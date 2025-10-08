const puppeteer = require('puppeteer');

async function wakeUpStreamlitApp() {
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--disable-gpu'
    ]
  });

  try {
    const page = await browser.newPage();
    
    // Set user agent to avoid detection
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');
    
    console.log('Accessing Streamlit app...');
    await page.goto('https://climviz.streamlit.app/', {
      waitUntil: 'networkidle2',
      timeout: 30000
    });

    // Wait a bit for the page to load
    await page.waitForTimeout(3000);

    // Check if the app is sleeping by looking for common sleep indicators
    const isSleeping = await page.evaluate(() => {
      const bodyText = document.body.innerText.toLowerCase();
      const sleepIndicators = [
        'your app is sleeping',
        'app is sleeping',
        'wake up',
        'click to wake up',
        'app is hibernating',
        'hibernating',
        'sleeping'
      ];
      
      return sleepIndicators.some(indicator => bodyText.includes(indicator));
    });

    if (isSleeping) {
      console.log('⚠️ App is sleeping, attempting to wake up...');
      
      // Look for wake up button or any clickable element that might wake the app
      const wakeUpButton = await page.$('button:contains("Wake up"), button:contains("wake up"), [data-testid*="wake"], [data-testid*="wake-up"]');
      
      if (wakeUpButton) {
        await wakeUpButton.click();
        console.log('✅ Wake up button clicked');
        await page.waitForTimeout(5000); // Wait for app to wake up
      } else {
        // If no specific wake up button, try clicking anywhere on the page
        console.log('No specific wake up button found, trying to click on page...');
        await page.click('body');
        await page.waitForTimeout(3000);
      }
      
      // Verify if the app is now awake
      const isStillSleeping = await page.evaluate(() => {
        const bodyText = document.body.innerText.toLowerCase();
        const sleepIndicators = [
          'your app is sleeping',
          'app is sleeping',
          'wake up',
          'click to wake up',
          'app is hibernating',
          'hibernating',
          'sleeping'
        ];
        
        return sleepIndicators.some(indicator => bodyText.includes(indicator));
      });

      if (!isStillSleeping) {
        console.log('✅ App successfully awakened!');
      } else {
        console.log('⚠️ App may still be sleeping, but wake up attempt was made');
      }
    } else {
      console.log('✅ App is already active and running');
    }

    // Take a screenshot for debugging (optional)
    await page.screenshot({ path: 'app_status.png', fullPage: true });
    console.log('📸 Screenshot saved as app_status.png');

  } catch (error) {
    console.error('❌ Error occurred:', error.message);
    throw error;
  } finally {
    await browser.close();
  }
}

// Run the function
wakeUpStreamlitApp()
  .then(() => {
    console.log('🎉 Wake up process completed successfully');
    process.exit(0);
  })
  .catch((error) => {
    console.error('💥 Wake up process failed:', error);
    process.exit(1);
  });
