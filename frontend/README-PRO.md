# DSPy Enterprise Platform - Professional UI

## Overview

Professional business-grade UI for DSPy framework with clean corporate design, no emojis, and enterprise-level functionality.

## Design Philosophy

**Corporate & Professional:**
- Corporate blue color scheme (#0066CC primary)
- Clean, minimalist design
- No emojis or playful elements
- Business-focused aesthetics
- Professional typography and spacing

**Functional:**
- All features from enhanced UI
- Educational content from original UI
- Production-ready components
- Real-time monitoring

## Color Palette

- **Primary Blue:** #0066CC (Corporate)
- **Dark Gray:** #2C3E50 (Sidebar, Headers)
- **Success Green:** #10B981
- **Warning Orange:** #F59E0B
- **Background:** #F8F9FA (Light Gray)
- **Borders:** Various gray shades for hierarchy

## Features

### 1. Dashboard
- Real-time statistics (requests, latency, success rate, active models)
- Recent activity feed
- Active models display
- Platform overview with key features

### 2. Module Testing
- Question Answering (QA)
- RAG Search
- Text Classification
- Full result display with reasoning
- Performance metrics
- Execution logs

### 3. Learn DSPy
**Educational content including:**
- What DSPy optimizes (instructions, examples, reasoning)
- Before/After optimization comparison
- 5-step optimization workflow
- Signature examples with code
- Module architecture examples
- Real code samples

### 4. Request History
- Automatic saving to localStorage
- Searchable history (last 100 requests)
- Click to reload past queries
- Performance metrics per request
- Clear history option

### 5. Model Artifacts
- List all compiled programs
- File details (size, type, date)
- Hot-swap between versions
- Version activation without restart
- Active/inactive status indicators

### 6. Settings
- Model selection (GPT-4o, GPT-4o-mini, Claude)
- Temperature slider with live value
- Max tokens configuration
- Optimizer selection (MIPRO, Bootstrap, Copro)
- Settings persistence in localStorage
- JSON preview of current config

## Quick Start

```bash
# 1. Start API
cd /Users/artemk/dspy-optimization-patterns
make run-api

# 2. Open UI
open frontend/index-pro.html
```

## Navigation

**Main Section:**
- Dashboard - Overview and statistics
- Testing - Interactive module testing
- Learn DSPy - Educational content
- History - Past requests

**Management Section:**
- Artifacts - Compiled program management
- Settings - Configuration

## Key Improvements

### vs Original UI
- Added all educational content
- Integrated code examples
- Step-by-step optimization guide
- Professional design system

### vs Enhanced UI
- Removed all emojis
- Corporate color scheme (blue instead of purple)
- Cleaner, more business-focused design
- Better visual hierarchy
- Professional badges and labels

## Technical Details

**Built with:**
- React 18 (via CDN)
- No external dependencies
- Vanilla CSS with CSS variables
- LocalStorage for persistence
- Responsive grid layouts

**Browser Compatibility:**
- Chrome/Edge (recommended)
- Firefox
- Safari
- All modern browsers with React 18 support

## API Integration

Connects to localhost:8000 with these endpoints:

- `GET /health` - API status check
- `POST /qa` - Question answering
- `POST /rag` - RAG search
- `POST /classify` - Classification
- `GET /artifacts` - List compiled programs
- `POST /artifacts/{name}/activate` - Switch model version
- `GET /stats` - System statistics

## File Structure

```
frontend/
├── index.html         # Original basic UI
├── index-enhanced.html # Enhanced with emojis & purple
├── index-pro.html     # Professional business UI ⭐
├── README-PRO.md      # This file
└── ENHANCED_UI_GUIDE.md
```

## Usage Examples

### Testing a Module

1. Click "Testing" in sidebar
2. Choose QA, RAG, or Classification
3. Fill in fields
4. Click "Run Test"
5. View results with reasoning and metrics

### Switching Model Versions

1. Click "Artifacts" in sidebar
2. See list of compiled programs
3. Click on a version
4. Click "Activate This Version"
5. API switches immediately

### Learning DSPy

1. Click "Learn DSPy" in sidebar
2. Read about optimization
3. See before/after examples
4. Review signature definitions
5. Study module architecture

## Customization

### Changing Colors

Edit CSS variables at the top of the file:

```css
:root {
    --primary: #0066CC;  /* Change primary color */
    --secondary: #2C3E50;  /* Change sidebar color */
    /* etc. */
}
```

### Adding New Sections

1. Create `renderNewSection()` function
2. Add nav item in sidebar
3. Add condition in main render:
   ```jsx
   {activeTab === 'newsection' && renderNewSection()}
   ```

## Performance

- **Lightweight:** Single HTML file, ~130KB
- **Fast:** No build step, loads instantly
- **Efficient:** React production build via CDN
- **Scalable:** Handles 100+ history items smoothly

## Troubleshooting

**API shows "Offline":**
- Check if API is running: `curl localhost:8000/health`
- Start API: `make run-api`

**No artifacts showing:**
- Run optimization: `make optimize-rag`
- Check `artifacts/compiled_programs/` directory

**History not saving:**
- Check browser localStorage (F12 → Application → Local Storage)
- Ensure not in Incognito/Private mode

**Settings reset:**
- Click "Save Settings" button after changes
- Check localStorage not blocked by browser

## Best Practices

1. **Regular Optimization:** Run `make optimize-rag` weekly with new data
2. **Monitor Dashboard:** Check latency and success rates daily
3. **Version Control:** Keep track of which artifacts work best
4. **History Management:** Clear history monthly to keep UI fast
5. **Settings Backup:** Export settings JSON before major changes

## Enterprise Features

- Professional color scheme suitable for corporate environments
- No distracting emojis or playful elements
- Clean, hierarchical information architecture
- Consistent spacing and typography
- Business-appropriate labels and badges
- Production-ready error handling
- Comprehensive monitoring capabilities

## Support

For issues or questions:
1. Check this README
2. Review main project README
3. Check QUICKSTART.md
4. Open GitHub issue

## Version

**v1.0.0** - Professional Enterprise UI
- Clean business design
- No emojis
- Corporate color scheme
- Full feature parity with enhanced UI
- Educational content integrated

---

**Built for DSPy Production Framework**
Professional UI for enterprise LLM applications
