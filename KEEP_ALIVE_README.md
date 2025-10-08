# Streamlit App Keep-Alive Feature

## Overview
To prevent Streamlit Community Cloud from going into sleep mode after 12 hours of inactivity, this project uses an intelligent browser automation solution that can detect when the app is sleeping and automatically wake it up by simulating user interactions.

## Configuration Details

### 1. Intelligent Wake-Up System
- **Browser Automation**: Uses Puppeteer to control a headless browser
- **Sleep Detection**: Automatically detects when the app is sleeping
- **Wake-Up Simulation**: Simulates clicking wake-up buttons or page interactions
- **Screenshot Capture**: Takes screenshots for debugging and verification

### 2. GitHub Actions Workflow
- File location: `.github/workflows/keep_alive.yml`
- Access frequency: Every 8 hours
- Trigger methods: Scheduled trigger + Manual trigger
- Dependencies: Node.js 18 + Puppeteer

### 3. How to Modify App URL
If you need to modify the Streamlit app URL, edit the following line in `wake_up_app.js`:

```javascript
await page.goto('https://your-app-url.streamlit.app/', {
```

Replace `https://your-app-url.streamlit.app` with your actual app URL.

### 4. How the Intelligent Wake-Up Works
The system uses the following process:

1. **Launch Headless Browser**: Starts a Puppeteer-controlled browser
2. **Navigate to App**: Visits your Streamlit app URL
3. **Detect Sleep State**: Scans page content for sleep indicators like:
   - "Your app is sleeping"
   - "Wake up" buttons
   - "App is hibernating"
4. **Wake Up Action**: If sleeping, attempts to:
   - Click specific wake-up buttons
   - Simulate user interaction by clicking on the page
5. **Verify Success**: Checks if the app is now awake
6. **Screenshot**: Captures a screenshot for debugging

### 5. How to Adjust Access Frequency
Modify the cron expression to adjust access frequency:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
  - cron: '0 */4 * * *'  # Every 4 hours
  - cron: '0 */8 * * *'  # Every 8 hours
  - cron: '0 0 * * *'    # Daily at midnight
```

### 6. How to Enable the Feature
1. Ensure your code is pushed to the GitHub repository
2. In the GitHub repository, go to the "Actions" tab
3. Find the "Keep Streamlit App Alive" workflow
4. Click "Run workflow" to test manually once

### 7. Monitor Status
- View workflow execution history on the GitHub Actions page
- Green checkmark indicates success
- Red X indicates failure, check logs for troubleshooting

## Important Notes
- This feature will periodically access your app, ensure it doesn't impact app performance
- Recommended access frequency is not too frequent, every 6 hours is sufficient to prevent sleep
- If the app URL changes, update the workflow configuration promptly

## Troubleshooting

### Common Status Codes
- **200**: App is running normally
- **303/302**: Redirect response (normal for Streamlit Community Cloud due to authentication)
- **000**: Network error or app is down
- **404**: App not found or URL is incorrect

### If the workflow execution fails, check:
1. Whether the app URL is correct
2. Whether the app is running normally
3. Whether the network connection is normal
4. Whether GitHub Actions is enabled

### About HTTP 303 Status Code
The 303 status code is **normal** for Streamlit Community Cloud apps. It indicates that the app is running but requires authentication, which causes redirects. The workflow is designed to handle this correctly and will consider 303 responses as successful app access.

### About Exit Code 47
Exit code 47 typically occurs when curl encounters SSL/TLS issues or infinite redirect loops. The workflow has been updated to:
- Not follow redirects automatically (avoiding infinite loops)
- Use appropriate timeout settings
- Handle redirect responses as successful app access

This ensures the keep-alive function works reliably with Streamlit Community Cloud's authentication system.
