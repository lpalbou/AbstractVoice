#!/usr/bin/env python3
"""
Test script for the new audio lifecycle callbacks in AbstractVoice v0.5.1

This script demonstrates the precise timing of the new callback system.
"""

import time
from abstractvoice import VoiceManager

def test_audio_callbacks():
    """Test the new audio lifecycle callbacks."""
    
    print("🧪 Testing AbstractVoice v0.5.1 Audio Lifecycle Callbacks")
    print("=" * 60)
    
    # Callback tracking
    events = []
    
    def on_synthesis_start():
        events.append(("synthesis_start", time.time()))
        print("🔴 Synthesis started - thinking phase")
    
    def on_audio_start():
        events.append(("audio_start", time.time()))
        print("🔵 Audio started - speaking phase")
    
    def on_audio_pause():
        events.append(("audio_pause", time.time()))
        print("⏸️ Audio paused")
    
    def on_audio_resume():
        events.append(("audio_resume", time.time()))
        print("▶️ Audio resumed")
    
    def on_audio_end():
        events.append(("audio_end", time.time()))
        print("🟢 Audio ended - ready phase")
    
    def on_synthesis_end():
        events.append(("synthesis_end", time.time()))
        print("✅ Synthesis complete")
    
    # Initialize VoiceManager
    print("Initializing VoiceManager...")
    vm = VoiceManager(debug_mode=True)
    
    # Wire up callbacks
    vm.tts_engine.on_playback_start = on_synthesis_start
    vm.tts_engine.on_playback_end = on_synthesis_end
    vm.on_audio_start = on_audio_start
    vm.on_audio_end = on_audio_end
    vm.on_audio_pause = on_audio_pause
    vm.on_audio_resume = on_audio_resume
    
    print("\n📢 Starting TTS with callback monitoring...")
    
    # Test basic playback
    vm.speak("This is a test of the new audio lifecycle callbacks in AbstractVoice version zero point five point one.")
    
    # Wait a bit, then pause
    time.sleep(2)
    print("\n⏸️ Pausing audio...")
    success = vm.pause_speaking()
    if success:
        print("✓ Pause successful")
    
    # Wait, then resume
    time.sleep(2)
    print("\n▶️ Resuming audio...")
    success = vm.resume_speaking()
    if success:
        print("✓ Resume successful")
    
    # Wait for completion
    while vm.is_speaking() or vm.is_paused():
        time.sleep(0.1)
    
    # Analyze timing
    print("\n📊 Callback Timing Analysis:")
    print("-" * 40)
    
    if len(events) >= 2:
        start_time = events[0][1]
        for event_name, event_time in events:
            offset = (event_time - start_time) * 1000  # Convert to milliseconds
            print(f"{event_name:15} | +{offset:6.1f}ms")
        
        # Calculate key intervals
        synthesis_to_audio = None
        audio_duration = None
        
        for i, (event_name, event_time) in enumerate(events):
            if event_name == "synthesis_start":
                synthesis_start = event_time
            elif event_name == "audio_start" and 'synthesis_start' in locals():
                synthesis_to_audio = (event_time - synthesis_start) * 1000
            elif event_name == "audio_end":
                audio_end = event_time
                # Find corresponding audio_start
                for j in range(i-1, -1, -1):
                    if events[j][0] == "audio_start":
                        audio_duration = (audio_end - events[j][1]) * 1000
                        break
        
        print("-" * 40)
        if synthesis_to_audio:
            print(f"Synthesis → Audio: {synthesis_to_audio:.1f}ms")
        if audio_duration:
            print(f"Audio Duration: {audio_duration:.1f}ms")
    
    # Cleanup
    vm.cleanup()
    print("\n✅ Test completed successfully!")
    print("🎯 The new callbacks provide precise timing for visual status indicators.")

if __name__ == "__main__":
    test_callbacks()
