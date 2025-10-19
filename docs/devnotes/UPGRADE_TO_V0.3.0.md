# AbstractVoice v0.3.0 - Major CLI Simplification

## 🎉 **Major Changes**

### ✅ **Unified CLI Structure**
**Before**: Confusing dual entry points
```bash
abstractvoice               # Voice mode only
abstractvoice-cli cli       # CLI REPL
abstractvoice-cli web       # Web API
abstractvoice-cli simple    # Simple demo
abstractvoice-cli check-deps # Dependency check
```

**After**: Single unified command
```bash
abstractvoice               # Voice mode (default)
abstractvoice cli           # CLI REPL
abstractvoice web           # Web API
abstractvoice simple        # Simple demo
abstractvoice check-deps    # Dependency check
```

### ✅ **Removed Redundancy**
- **BREAKING CHANGE**: `abstractvoice-cli` command removed
- All functionality moved to single `abstractvoice` command
- Cleaner, more intuitive user experience

## 🔄 **Migration Guide**

### **For Users**
Replace `abstractvoice-cli` with `abstractvoice`:

```bash
# Old commands (no longer work)
abstractvoice-cli cli --language fr
abstractvoice-cli web
abstractvoice-cli simple
abstractvoice-cli check-deps

# New commands (v0.3.0+)
abstractvoice cli --language fr
abstractvoice web
abstractvoice simple
abstractvoice check-deps
```

### **For Scripts/Automation**
Update any scripts that use `abstractvoice-cli`:

```bash
# Old
#!/bin/bash
abstractvoice-cli check-deps

# New
#!/bin/bash
abstractvoice check-deps
```

## 📋 **Complete Command Reference**

### **Voice Mode (Default)**
```bash
abstractvoice                              # Interactive voice mode
abstractvoice --model cogito:3b            # With custom Ollama model
abstractvoice --language fr                # French voice mode
abstractvoice --temperature 0.7            # Custom AI parameters
abstractvoice --no-listening               # TTS only (no STT)
```

### **Examples & Utilities**
```bash
abstractvoice cli                          # CLI REPL for text interaction
abstractvoice cli --language de            # German CLI REPL
abstractvoice web                          # Start web API server
abstractvoice simple                       # Run simple demo
abstractvoice check-deps                   # Check dependency compatibility
abstractvoice help                         # Show available commands
```

### **Get Help**
```bash
abstractvoice --help                       # Full help with all options
abstractvoice help                         # Show available commands
```

## ⚡ **Additional Improvements in v0.3.0**

### **Enhanced Dependency Management**
- Fixed PyTorch/TorchVision conflicts (`RuntimeError: operator torchvision::nms does not exist`)
- Added comprehensive dependency checker
- Restructured optional dependency groups for clarity

### **Better Error Messages**
- Context-aware error handling for Ollama/model issues
- Specific guidance for connection errors and missing models
- Actionable error messages with exact resolution commands

### **Documentation Updates**
- All documentation updated to use unified `abstractvoice` command
- Enhanced installation guides with troubleshooting
- Simplified CLI reference

## 🧪 **Testing Your Upgrade**

### **1. Check Version**
```bash
python -c "import abstractvoice; print(abstractvoice.__version__)"
# Should show: 0.3.0
```

### **2. Test Unified CLI**
```bash
abstractvoice --help                       # Should show unified interface
abstractvoice check-deps                   # Should work
abstractvoice help                         # Should show examples
```

### **3. Verify Old Commands Don't Work**
```bash
abstractvoice-cli --help                   # Should fail (command not found)
```

## 📝 **What This Means**

### **✅ Benefits**
- **Simpler**: One command to remember instead of two
- **Cleaner**: No confusion about which command to use
- **Consistent**: All functionality through single entry point
- **Intuitive**: `abstractvoice <what-you-want>` format

### **⚠️ Breaking Changes**
- **`abstractvoice-cli` command removed** (functionality moved to `abstractvoice`)
- **Scripts using `abstractvoice-cli` need updating**
- **Documentation/tutorials referencing `abstractvoice-cli` are outdated**

## 🚀 **Upgrade Instructions**

### **For End Users**
```bash
# 1. Upgrade to v0.3.0
pip install --upgrade abstractvoice

# 2. Update your muscle memory
abstractvoice check-deps                   # Not abstractvoice-cli check-deps
abstractvoice cli                          # Not abstractvoice-cli cli

# 3. Test everything works
abstractvoice --help
```

### **For Developers/Scripts**
1. **Update scripts**: Replace `abstractvoice-cli` with `abstractvoice`
2. **Update documentation**: Remove `abstractvoice-cli` references
3. **Update CI/CD**: Update any automated scripts
4. **Test thoroughly**: Ensure all functionality still works

## 📊 **Version Comparison**

| Feature | v0.2.x | v0.3.0 |
|---------|--------|--------|
| Voice Mode | `abstractvoice` | `abstractvoice` ✅ |
| CLI REPL | `abstractvoice-cli cli` | `abstractvoice cli` ✅ |
| Web API | `abstractvoice-cli web` | `abstractvoice web` ✅ |
| Simple Demo | `abstractvoice-cli simple` | `abstractvoice simple` ✅ |
| Check Deps | `abstractvoice-cli check-deps` | `abstractvoice check-deps` ✅ |
| Entry Points | 2 (confusing) | 1 (clean) ✅ |

---

**The CLI is now simpler, cleaner, and more intuitive! 🎉**

All functionality is preserved - just accessed through the unified `abstractvoice` command.