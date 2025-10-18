"""TTS Engine for high-quality speech synthesis with interrupt handling.

This module implements best practices for TTS synthesis including:
- Sentence segmentation for long text (prevents attention degradation)
- Text chunking for extremely long content
- Text preprocessing and normalization
- Robust error handling
"""

import threading
import time
import numpy as np
import sounddevice as sd
import os
import sys
import logging
import warnings
import re
from TTS.api import TTS
import librosa
import queue

# Suppress the PyTorch FutureWarning about torch.load
warnings.filterwarnings(
    "ignore", 
    message="You are using `torch.load` with `weights_only=False`", 
    category=FutureWarning
)

# Suppress pkg_resources deprecation warning from jieba
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*",
    category=DeprecationWarning
)

# Suppress coqpit deserialization warnings from TTS models
warnings.filterwarnings(
    "ignore",
    message=".*Type mismatch.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*Failed to deserialize field.*",
    category=UserWarning
)

# Suppress macOS audio warnings (harmless but annoying)
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

def preprocess_text(text):
    """Preprocess text for better TTS synthesis.
    
    This function normalizes text to prevent synthesis errors:
    - Removes excessive whitespace
    - Normalizes punctuation
    - Handles common abbreviations
    - Removes problematic characters
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return text
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize ellipsis
    text = text.replace('...', '.')
    
    # Remove or normalize problematic characters
    # Keep basic punctuation that helps with prosody
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
    
    return text.strip()


def apply_speed_without_pitch_change(audio, speed, sr=22050):
    """Apply speed change without affecting pitch using librosa time_stretch.
    
    Args:
        audio: Audio samples as numpy array
        speed: Speed multiplier (0.5-2.0, where >1.0 is faster, <1.0 is slower)
        sr: Sample rate (default 22050)
        
    Returns:
        Time-stretched audio samples
    """
    if speed == 1.0:
        return audio
    
    # librosa.effects.time_stretch expects rate parameter where:
    # rate > 1.0 makes audio faster (shorter)
    # rate < 1.0 makes audio slower (longer)
    # This matches our speed semantics
    try:
        stretched_audio = librosa.effects.time_stretch(audio, rate=speed)
        return stretched_audio
    except Exception as e:
        # If time-stretching fails, return original audio
        logging.warning(f"Time-stretching failed: {e}, using original audio")
        return audio


class NonBlockingAudioPlayer:
    """Non-blocking audio player using OutputStream callbacks for immediate pause/resume."""
    
    def __init__(self, sample_rate=22050, debug_mode=False):
        self.sample_rate = sample_rate
        self.debug_mode = debug_mode
        
        # Audio queue and playback state
        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.pause_lock = threading.Lock()
        
        # Current audio buffer management
        self.current_audio = None
        self.current_position = 0
        self.playback_complete_callback = None
        
    def _audio_callback(self, outdata, frames, time, status):
        """Callback function for OutputStream - provides immediate pause/resume."""
        if status and self.debug_mode:
            print(f"Audio callback status: {status}")
        
        # Check pause state (thread-safe)
        with self.pause_lock:
            if self.is_paused:
                # Output silence when paused - immediate response
                outdata.fill(0)
                return
        
        try:
            # Get next audio chunk if needed
            if self.current_audio is None or self.current_position >= len(self.current_audio):
                try:
                    self.current_audio = self.audio_queue.get_nowait()
                    self.current_position = 0
                    if self.debug_mode:
                        print(f" > Playing audio chunk ({len(self.current_audio)} samples)")
                except queue.Empty:
                    # No more audio - output silence and mark as not playing
                    outdata.fill(0)
                    if self.is_playing:
                        self.is_playing = False
                        if self.playback_complete_callback:
                            # Call completion callback in a separate thread to avoid blocking
                            threading.Thread(target=self.playback_complete_callback, daemon=True).start()
                    return
            
            # Calculate how much audio we can output this frame
            remaining = len(self.current_audio) - self.current_position
            frames_to_output = min(frames, remaining)
            
            # Output the audio data
            if frames_to_output > 0:
                # Handle both mono and stereo output
                if outdata.shape[1] == 1:  # Mono output
                    outdata[:frames_to_output, 0] = self.current_audio[self.current_position:self.current_position + frames_to_output]
                else:  # Stereo output
                    audio_data = self.current_audio[self.current_position:self.current_position + frames_to_output]
                    outdata[:frames_to_output, 0] = audio_data  # Left channel
                    outdata[:frames_to_output, 1] = audio_data  # Right channel
                
                self.current_position += frames_to_output
            
            # Fill remaining with silence if needed
            if frames_to_output < frames:
                outdata[frames_to_output:].fill(0)
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error in audio callback: {e}")
            outdata.fill(0)
    
    def start_stream(self):
        """Start the audio stream."""
        if self.stream is None:
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,  # Mono output
                    callback=self._audio_callback,
                    blocksize=1024,  # Small buffer for low latency
                    dtype=np.float32
                )
                self.stream.start()
                if self.debug_mode:
                    print(" > Audio stream started")
            except Exception as e:
                if self.debug_mode:
                    print(f"Error starting audio stream: {e}")
                raise
    
    def stop_stream(self):
        """Stop the audio stream."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                if self.debug_mode:
                    print(" > Audio stream stopped")
            except Exception as e:
                if self.debug_mode:
                    print(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
        
        self.is_playing = False
        with self.pause_lock:
            self.is_paused = False
        self.clear_queue()
    
    def play_audio(self, audio_array):
        """Add audio to the playback queue."""
        if audio_array is not None and len(audio_array) > 0:
            # Ensure audio is float32 and normalized
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            self.audio_queue.put(audio_array)
            self.is_playing = True
            
            # Start stream if not already running
            if self.stream is None:
                self.start_stream()
    
    def pause(self):
        """Pause audio playback immediately."""
        with self.pause_lock:
            if self.is_playing and not self.is_paused:
                self.is_paused = True
                if self.debug_mode:
                    print(" > Audio paused immediately")
                return True
        return False
    
    def resume(self):
        """Resume audio playback immediately."""
        with self.pause_lock:
            if self.is_paused:
                self.is_paused = False
                if self.debug_mode:
                    print(" > Audio resumed immediately")
                return True
        return False
    
    def is_paused_state(self):
        """Check if audio is currently paused."""
        with self.pause_lock:
            return self.is_paused
    
    def clear_queue(self):
        """Clear the audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset current audio buffer
        self.current_audio = None
        self.current_position = 0


def chunk_long_text(text, max_chunk_size=300):
    """Split very long text into manageable chunks at natural boundaries.
    
    For extremely long texts, this function splits at paragraph or sentence
    boundaries to prevent memory issues and attention degradation.
    
    Args:
        text: Input text string
        max_chunk_size: Maximum characters per chunk (default 300)
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    
    # First try to split by paragraphs
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        # If adding this paragraph would exceed limit and we have content
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        # If a single paragraph is too long, split by sentences
        if len(current_chunk) > max_chunk_size:
            # Split on sentence boundaries
            sentences = re.split(r'([.!?]+\s+)', current_chunk)
            temp_chunk = ""
            
            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                punct = sentences[i+1] if i+1 < len(sentences) else ""
                
                if len(temp_chunk) + len(sentence) + len(punct) > max_chunk_size and temp_chunk:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = sentence + punct
                else:
                    temp_chunk += sentence + punct
            
            current_chunk = temp_chunk
    
    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


class TTSEngine:
    """Text-to-speech engine with interrupt capability."""
    
    def __init__(self, model_name="tts_models/en/ljspeech/vits", debug_mode=False, streaming=True):
        """Initialize the TTS engine.
        
        Args:
            model_name: TTS model to use (default: vits - best quality, requires espeak-ng)
            debug_mode: Enable debug output
            streaming: Enable streaming playback (start playing while synthesizing remaining chunks)
        
        Note:
            VITS model (default) requires espeak-ng for best quality:
            - macOS: brew install espeak-ng
            - Linux: sudo apt-get install espeak-ng  
            - Windows: See installation guide in README
            
            If espeak-ng is not available, will auto-fallback to fast_pitch
        """
        # Set up debug mode
        self.debug_mode = debug_mode
        self.streaming = streaming
        
        # Callback to notify when TTS starts/stops (for pausing voice recognition)
        self.on_playback_start = None
        self.on_playback_end = None
        
        # Suppress TTS output unless in debug mode
        if not debug_mode:
            # Suppress all TTS logging
            logging.getLogger('TTS').setLevel(logging.ERROR)
            logging.getLogger('TTS.utils.audio').setLevel(logging.ERROR)
            logging.getLogger('TTS.utils.io').setLevel(logging.ERROR)
            logging.getLogger('numba').setLevel(logging.ERROR)
            
            # Disable stdout during TTS loading
            os.environ['TTS_VERBOSE'] = '0'
            
            # Temporarily redirect stdout to suppress TTS init messages
            orig_stdout = sys.stdout
            null_out = open(os.devnull, 'w')
            sys.stdout = null_out
        
        try:
            if self.debug_mode:
                print(f" > Loading TTS model: {model_name}")
            
            # Try to initialize TTS
            try:
                self.tts = TTS(model_name=model_name, progress_bar=self.debug_mode)
            except Exception as e:
                error_msg = str(e).lower()
                # Check if this is an espeak-related error
                if ("espeak" in error_msg or "phoneme" in error_msg):
                    # Restore stdout to show user-friendly message
                    if not debug_mode:
                        sys.stdout = sys.__stdout__
                    
                    print("\n" + "="*70)
                    print("⚠️  VITS Model Requires espeak-ng (Not Found)")
                    print("="*70)
                    print("\nFor BEST voice quality, install espeak-ng:")
                    print("  • macOS:   brew install espeak-ng")
                    print("  • Linux:   sudo apt-get install espeak-ng")
                    print("  • Windows: conda install espeak-ng  (or see README)")
                    print("\nFalling back to fast_pitch (lower quality, but works)")
                    print("="*70 + "\n")
                    
                    if not debug_mode:
                        sys.stdout = null_out
                    
                    # Fallback to fast_pitch
                    self.tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=self.debug_mode)
                else:
                    # Different error, re-raise
                    raise
        finally:
            # Restore stdout if we redirected it
            if not debug_mode:
                sys.stdout = orig_stdout
                null_out.close()
        
        # Initialize non-blocking audio player for immediate pause/resume
        self.audio_player = NonBlockingAudioPlayer(sample_rate=22050, debug_mode=debug_mode)
        self.audio_player.playback_complete_callback = self._on_playback_complete
        
        # Legacy playback state (for compatibility with existing code)
        self.is_playing = False
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.pause_flag.set()  # Initially not paused (set means "not paused")
        self.playback_thread = None
        self.start_time = 0
        self.audio_queue = []  # Queue for streaming playback
        self.queue_lock = threading.Lock()  # Thread-safe queue access
        
        # Pause/resume state
        self.pause_lock = threading.Lock()  # Thread-safe pause operations
        self.is_paused_state = False  # Explicit paused state tracking
    
    def _on_playback_complete(self):
        """Callback when audio playback completes."""
        self.is_playing = False
        if self.on_playback_end:
            self.on_playback_end()
    
    def _speak_with_nonblocking_player(self, text, speed=1.0, callback=None):
        """Alternative speak method using NonBlockingAudioPlayer for immediate pause/resume."""
        # Stop any existing playback
        self.stop()
        
        if not text:
            return False
        
        try:
            # Preprocess text for better synthesis quality
            processed_text = preprocess_text(text)
            
            if self.debug_mode:
                print(f" > Speaking (non-blocking): '{processed_text[:100]}{'...' if len(processed_text) > 100 else ''}'")
                print(f" > Text length: {len(processed_text)} chars")
                if speed != 1.0:
                    print(f" > Using speed multiplier: {speed}x")
            
            # For very long text, chunk it at natural boundaries
            text_chunks = chunk_long_text(processed_text, max_chunk_size=300)
            
            if self.debug_mode and len(text_chunks) > 1:
                print(f" > Split into {len(text_chunks)} chunks for processing")
            
            # Set playing state
            self.is_playing = True
            self.is_paused_state = False
            
            # Call start callback
            if self.on_playback_start:
                self.on_playback_start()
            
            # Synthesize and queue audio chunks
            def synthesis_worker():
                try:
                    for i, chunk in enumerate(text_chunks):
                        if self.stop_flag.is_set():
                            break
                        
                        if self.debug_mode and len(text_chunks) > 1:
                            print(f" > Processing chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)...")
                        
                        # Generate audio for this chunk
                        chunk_audio = self.tts.tts(chunk, split_sentences=True)
                        
                        if chunk_audio and len(chunk_audio) > 0:
                            # Apply speed adjustment
                            if speed != 1.0:
                                chunk_audio = apply_speed_without_pitch_change(
                                    np.array(chunk_audio), speed
                                )
                            
                            # Queue the audio for playback
                            self.audio_player.play_audio(np.array(chunk_audio))
                            
                            if self.debug_mode:
                                print(f" > Chunk {i+1} queued ({len(chunk_audio)} samples)")
                        
                        # Small delay between chunks to prevent overwhelming the queue
                        time.sleep(0.01)
                
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error in synthesis worker: {e}")
                finally:
                    # Synthesis complete - audio player will handle completion callback
                    pass
            
            # Start synthesis in background thread
            synthesis_thread = threading.Thread(target=synthesis_worker, daemon=True)
            synthesis_thread.start()
            
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in _speak_with_nonblocking_player: {e}")
            self.is_playing = False
            return False
    
    def speak(self, text, speed=1.0, callback=None):
        """Convert text to speech and play audio.
        
        Implements SOTA best practices for long text synthesis:
        - Text preprocessing and normalization
        - Intelligent chunking for very long text (>500 chars)
        - Sentence segmentation to prevent attention degradation
        - Seamless audio concatenation for chunks
        
        Args:
            text: Text to convert to speech
            speed: Speed multiplier (0.5-2.0)
            callback: Function to call when speech is complete
        
        Returns:
            True if speech started, False if text was empty
        """
        # Use the new non-blocking audio player for immediate pause/resume
        return self._speak_with_nonblocking_player(text, speed, callback)
        
        if not text:
            return False
        
        try:
            # Preprocess text for better synthesis quality
            processed_text = preprocess_text(text)
            
            if self.debug_mode:
                print(f" > Speaking: '{processed_text[:100]}{'...' if len(processed_text) > 100 else ''}'")
                print(f" > Text length: {len(processed_text)} chars")
                if speed != 1.0:
                    print(f" > Using speed multiplier: {speed}x")
            
            # For very long text, chunk it at natural boundaries
            # Use 300 chars to stay well within model's training distribution
            text_chunks = chunk_long_text(processed_text, max_chunk_size=300)
            
            if self.debug_mode and len(text_chunks) > 1:
                print(f" > Split into {len(text_chunks)} chunks for processing")
            
            # Redirect stdout for non-debug mode
            orig_stdout = None
            null_out = None
            if not self.debug_mode:
                orig_stdout = sys.stdout
                null_out = open(os.devnull, 'w')
                sys.stdout = null_out
            
            try:
                # Choose synthesis strategy based on streaming mode
                if self.streaming and len(text_chunks) > 1:
                    # STREAMING MODE: Synthesize and play progressively
                    if self.debug_mode:
                        sys.stdout = sys.__stdout__
                        print(f" > Streaming mode: will start playback after first chunk")
                        if not self.debug_mode:
                            sys.stdout = null_out
                    
                    # Synthesize first chunk
                    if self.debug_mode:
                        sys.stdout = sys.__stdout__
                        print(f" > Processing chunk 1/{len(text_chunks)} ({len(text_chunks[0])} chars)...")
                        if not self.debug_mode:
                            sys.stdout = null_out
                    
                    first_audio = self.tts.tts(text_chunks[0], split_sentences=True)
                    
                    if not first_audio:
                        if self.debug_mode:
                            sys.stdout = sys.__stdout__
                            print("TTS failed to generate audio for first chunk.")
                        return False
                    
                    # Apply speed adjustment using time-stretching (preserves pitch)
                    if speed != 1.0:
                        first_audio = apply_speed_without_pitch_change(
                            np.array(first_audio), speed
                        )
                    
                    if self.debug_mode:
                        sys.stdout = sys.__stdout__
                        print(f" > Chunk 1 generated {len(first_audio)} audio samples")
                        if speed != 1.0:
                            print(f" > Applied time-stretch: {speed}x (pitch preserved)")
                        print(f" > Starting playback while synthesizing remaining chunks...")
                        if not self.debug_mode:
                            sys.stdout = null_out
                    
                    # Initialize queue with first chunk
                    with self.queue_lock:
                        self.audio_queue = [first_audio]
                    
                    # Start playback thread (will play from queue)
                    audio = None  # Will use queue instead
                    
                else:
                    # NON-STREAMING MODE: Synthesize all chunks then play
                    audio_chunks = []
                    for i, chunk in enumerate(text_chunks):
                        if self.debug_mode and len(text_chunks) > 1:
                            sys.stdout = sys.__stdout__
                            print(f" > Processing chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)...")
                            if not self.debug_mode:
                                sys.stdout = null_out
                        
                        # Use split_sentences=True (SOTA best practice)
                        chunk_audio = self.tts.tts(chunk, split_sentences=True)
                        
                        if chunk_audio:
                            # Apply speed adjustment using time-stretching (preserves pitch)
                            if speed != 1.0:
                                chunk_audio = apply_speed_without_pitch_change(
                                    np.array(chunk_audio), speed
                                )
                            audio_chunks.append(chunk_audio)
                            if self.debug_mode and len(text_chunks) > 1:
                                sys.stdout = sys.__stdout__
                                print(f" > Chunk {i+1} generated {len(chunk_audio)} audio samples")
                                if not self.debug_mode:
                                    sys.stdout = null_out
                        elif self.debug_mode:
                            sys.stdout = sys.__stdout__
                            print(f" > Warning: Chunk {i+1} failed to generate audio")
                            if not self.debug_mode:
                                sys.stdout = null_out
                    
                    if not audio_chunks:
                        if self.debug_mode:
                            sys.stdout = sys.__stdout__
                            print("TTS failed to generate audio.")
                        return False
                    
                    # Concatenate audio arrays
                    if len(audio_chunks) == 1:
                        audio = audio_chunks[0]
                    else:
                        audio = np.concatenate(audio_chunks)
                        if self.debug_mode:
                            sys.stdout = sys.__stdout__
                            print(f" > Concatenated {len(audio_chunks)} chunks into {len(audio)} total audio samples")
                            if not self.debug_mode:
                                sys.stdout = null_out
                
            finally:
                # Restore stdout if we redirected it
                if not self.debug_mode and orig_stdout:
                    sys.stdout = orig_stdout
                    if null_out:
                        null_out.close()
            
            def _audio_playback():
                try:
                    self.is_playing = True
                    self.start_time = time.time()
                    
                    # Notify that playback is starting (to pause voice recognition)
                    if self.on_playback_start:
                        self.on_playback_start()
                    
                    # Use standard playback rate (speed is handled via time-stretching)
                    playback_rate = 22050
                    
                    # STREAMING MODE: Play from queue while synthesizing remaining chunks
                    if audio is None:  # Streaming mode indicator
                        # Start background thread to synthesize remaining chunks
                        def _synthesize_remaining():
                            for i in range(1, len(text_chunks)):
                                if self.stop_flag.is_set():
                                    break
                                
                                if self.debug_mode:
                                    print(f" > [Background] Processing chunk {i+1}/{len(text_chunks)} ({len(text_chunks[i])} chars)...")
                                
                                try:
                                    chunk_audio = self.tts.tts(text_chunks[i], split_sentences=True)
                                    if chunk_audio:
                                        # Apply speed adjustment using time-stretching (preserves pitch)
                                        if speed != 1.0:
                                            chunk_audio = apply_speed_without_pitch_change(
                                                np.array(chunk_audio), speed
                                            )
                                        with self.queue_lock:
                                            self.audio_queue.append(chunk_audio)
                                        if self.debug_mode:
                                            print(f" > [Background] Chunk {i+1} generated {len(chunk_audio)} samples, added to queue")
                                except Exception as e:
                                    if self.debug_mode:
                                        print(f" > [Background] Chunk {i+1} synthesis error: {e}")
                        
                        synthesis_thread = threading.Thread(target=_synthesize_remaining)
                        synthesis_thread.daemon = True
                        synthesis_thread.start()
                        
                        # Play chunks from queue as they become available
                        chunks_played = 0
                        while chunks_played < len(text_chunks) and not self.stop_flag.is_set():
                            # Check for pause before processing next chunk
                            while not self.pause_flag.is_set() and not self.stop_flag.is_set():
                                time.sleep(0.1)  # Non-blocking pause check
                            
                            if self.stop_flag.is_set():
                                break
                            
                            # Wait for next chunk to be available
                            while True:
                                with self.queue_lock:
                                    if chunks_played < len(self.audio_queue):
                                        chunk_to_play = self.audio_queue[chunks_played]
                                        break
                                if self.stop_flag.is_set():
                                    break
                                time.sleep(0.05)  # Short wait before checking again
                            
                            if self.stop_flag.is_set():
                                break
                            
                            # Play this chunk
                            audio_array = np.array(chunk_to_play)
                            sd.play(audio_array, samplerate=playback_rate)
                            
                            # Wait for this chunk to finish (with frequent pause checks)
                            while not self.stop_flag.is_set() and sd.get_stream().active:
                                # Check for pause more frequently
                                if not self.pause_flag.is_set():
                                    # Paused - let current audio finish naturally (avoids terminal interference)
                                    break
                                time.sleep(0.05)  # Check every 50ms for better responsiveness
                            
                            if self.stop_flag.is_set():
                                # Only use sd.stop() for explicit stop, not pause
                                sd.stop()
                                break
                            
                            chunks_played += 1
                        
                        synthesis_thread.join(timeout=1.0)  # Wait for synthesis to complete
                    
                    else:
                        # NON-STREAMING MODE: Play concatenated audio
                        audio_array = np.array(audio)
                        sd.play(audio_array, samplerate=playback_rate)
                        
                        # Wait for playback to complete or stop flag (with pause support)
                        while not self.stop_flag.is_set() and sd.get_stream().active:
                            # Check for pause more frequently
                            if not self.pause_flag.is_set():
                                # Paused - let current audio finish naturally and wait
                                if self.debug_mode:
                                    print(" > Audio paused, waiting for resume...")
                                # Non-blocking wait for resume
                                while not self.pause_flag.is_set() and not self.stop_flag.is_set():
                                    time.sleep(0.1)
                                if not self.stop_flag.is_set():
                                    # Resume - restart the audio (non-streaming limitation)
                                    if self.debug_mode:
                                        print(" > Resuming audio from beginning of current segment...")
                                    sd.play(audio_array, samplerate=playback_rate)
                            time.sleep(0.05)  # Check every 50ms for better responsiveness
                        
                        sd.stop()
                    
                    self.is_playing = False
                    
                    # Notify that playback has ended (to resume voice recognition)
                    if self.on_playback_end:
                        self.on_playback_end()
                    
                    if self.debug_mode:
                        duration = time.time() - self.start_time
                        if not self.stop_flag.is_set():  # Only if completed normally
                            print(f" > Speech completed in {duration:.2f} seconds")
                    
                    # Call the callback if provided and speech completed normally
                    if callback and not self.stop_flag.is_set():
                        callback()
                
                except Exception as e:
                    if self.debug_mode:
                        print(f"Audio playback error: {e}")
                    self.is_playing = False
                    # Ensure we notify end even on error
                    if self.on_playback_end:
                        self.on_playback_end()
            
            # Start playback in a separate thread
            self.stop_flag.clear()
            self.pause_flag.set()  # Ensure we start unpaused
            self.is_paused_state = False  # Reset paused state
            self.playback_thread = threading.Thread(target=_audio_playback)
            self.playback_thread.start()
            return True
        
        except Exception as e:
            if self.debug_mode:
                print(f"TTS error: {e}")
            return False
    
    def stop(self):
        """Stop current audio playback.
        
        Returns:
            True if playback was stopped, False if no playback was active
        """
        stopped = False
        
        # Stop new non-blocking audio player
        if self.audio_player.is_playing:
            self.audio_player.stop_stream()
            stopped = True
            if self.debug_mode:
                print(" > TTS playback stopped (non-blocking)")
        
        # Stop legacy playback system
        if self.playback_thread and self.playback_thread.is_alive():
            self.stop_flag.set()
            self.pause_flag.set()  # Ensure we're not stuck in pause
            self.is_paused_state = False  # Reset paused state
            self.playback_thread.join()
            self.playback_thread = None
            stopped = True
            
            if self.debug_mode:
                print(" > TTS playback interrupted (legacy)")
        
        # Reset state
        self.is_playing = False
        self.is_paused_state = False
        
        return stopped
    
    def pause(self):
        """Pause current speech playback.
        
        Uses a non-interfering pause method that avoids terminal I/O issues.
        
        Returns:
            True if paused, False if no playback was active
        """
        # Try new non-blocking audio player first
        if self.audio_player.is_playing:
            result = self.audio_player.pause()
            if result:
                self.is_paused_state = True
                if self.debug_mode:
                    print(" > TTS paused immediately (non-blocking)")
            return result
        
        # Fallback to legacy system
        if self.playback_thread and self.playback_thread.is_alive() and self.is_playing:
            self.pause_flag.clear()  # Clear means "paused"
            self.is_paused_state = True  # Explicit state tracking
            
            if self.debug_mode:
                print(" > TTS paused (legacy method)")
            
            return True
        
        return False
    
    def resume(self):
        """Resume paused speech playback.
        
        Returns:
            True if resumed, False if not paused or no playback active
        """
        if self.is_paused_state:
            # Try new non-blocking audio player first
            if self.audio_player.is_paused_state():
                result = self.audio_player.resume()
                if result:
                    self.is_paused_state = False
                    if self.debug_mode:
                        print(" > TTS resumed immediately (non-blocking)")
                    return True
            
            # Fallback to legacy system
            if self.playback_thread and self.playback_thread.is_alive():
                # Thread is still alive, can resume
                self.pause_flag.set()  # Set means "not paused"
                self.is_paused_state = False  # Clear explicit state
                if self.debug_mode:
                    print(" > TTS resumed (legacy method)")
                return True
            else:
                # Thread died while paused, nothing to resume
                self.is_paused_state = False  # Clear paused state
                if self.debug_mode:
                    print(" > TTS was paused but playback already completed")
                return False
        return False
    
    def is_paused(self):
        """Check if TTS is currently paused.
        
        Returns:
            True if paused, False otherwise
        """
        return self.is_paused_state
    
    def is_active(self):
        """Check if TTS is currently playing.
        
        Returns:
            True if TTS is active, False otherwise
        """
        return self.is_playing 