# Streamlit App Keep-Alive Feature

## Overview
To prevent Streamlit Community Cloud from going into sleep mode after 12 hours of inactivity, this project is configured with GitHub Actions scheduled access functionality.

## Configuration Details

### 1. GitHub Actions Workflow
- File location: `.github/workflows/keep_alive.yml`
- Access frequency: Every 6 hours
- Trigger methods: Scheduled trigger + Manual trigger

### 2. How to Modify App URL
If you need to modify the Streamlit app URL, edit the following line in `.github/workflows/keep_alive.yml`:

```yaml
response_code=$(curl -s -o /dev/null -w "%{http_code}" https://your-app-url.streamlit.app)
```

Replace `https://your-app-url.streamlit.app` with your actual app URL.

### 3. How to Adjust Access Frequency
Modify the cron expression to adjust access frequency:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
  - cron: '0 */4 * * *'  # Every 4 hours
  - cron: '0 */8 * * *'  # Every 8 hours
  - cron: '0 0 * * *'    # Daily at midnight
```

### 4. How to Enable the Feature
1. Ensure your code is pushed to the GitHub repository
2. In the GitHub repository, go to the "Actions" tab
3. Find the "Keep Streamlit App Alive" workflow
4. Click "Run workflow" to test manually once

### 5. Monitor Status
- View workflow execution history on the GitHub Actions page
- Green checkmark indicates success
- Red X indicates failure, check logs for troubleshooting

## Important Notes
- This feature will periodically access your app, ensure it doesn't impact app performance
- Recommended access frequency is not too frequent, every 6 hours is sufficient to prevent sleep
- If the app URL changes, update the workflow configuration promptly

## Troubleshooting
If the workflow execution fails, check:
1. Whether the app URL is correct
2. Whether the app is running normally
3. Whether the network connection is normal
4. Whether GitHub Actions is enabled
