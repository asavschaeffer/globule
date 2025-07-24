# Glass Engine Quick Start

*New to Globule? Start here to understand and try the system in 5 minutes.*

## What is the Glass Engine?

The Glass Engine is Globule's revolutionary tutorial system that **shows you exactly how the software works** while you learn to use it. No black boxes, no guesswork - complete transparency.

## 30-Second Start

1. **Install Globule:**
   ```bash
   pip install -e .  # from the globule directory
   ```

2. **Run your first tutorial:**
   ```bash
   globule tutorial --mode=interactive
   ```

3. **Follow the guided experience** - it will teach you everything you need to know!

## Which Mode Should I Choose?

### 🎓 **New User?** → `--mode=interactive`
- Guided hands-on tutorial
- Uses your own thoughts and ideas
- Explains every step
- Perfect for learning

### 🎪 **Want to see capabilities?** → `--mode=demo` 
- Professional technical showcase
- Automated examples
- Performance benchmarks
- Great for understanding potential

### 🔧 **Need technical details?** → `--mode=debug`
- Raw system data
- Complete execution traces  
- Maximum technical depth
- Perfect for debugging/analysis

## Example Commands

```bash
# Start with guided learning (recommended)
globule tutorial --mode=interactive

# See a professional demonstration
globule tutorial --mode=demo

# Get raw technical details
globule tutorial --mode=debug

# Get help with tutorial options
globule tutorial --help
```

## What You'll Learn

Every Glass Engine mode teaches you:

- ✅ **How Globule captures your thoughts** - complete data flow
- ✅ **Where your data is stored** - file locations and database structure  
- ✅ **How AI processes your ideas** - embedding and parsing details
- ✅ **How to find related thoughts** - retrieval and clustering
- ✅ **Privacy and control** - local-first architecture benefits

## After the Tutorial

Once you understand how Globule works:

```bash
# Start capturing your thoughts
globule add "Your brilliant idea here"

# Organize them into drafts  
globule draft "your topic"

# Run the tutorial anytime to refresh your understanding
globule tutorial --mode=interactive
```

## Need More Detail?

- **Full Guide:** [`docs/glass-engine-guide.md`](./glass-engine-guide.md)
- **Technical Documentation:** [`src/globule/tutorial/`](../src/globule/tutorial/)
- **Philosophy:** Glass Engine unifies tests, tutorials, and showcases into one transparent experience

## The Philosophy in One Sentence

> "Let the user see exactly how the pistons fire while teaching them to drive."

**Ready to begin?** Run `globule tutorial --mode=interactive` and start your journey into transparent software! 🚀