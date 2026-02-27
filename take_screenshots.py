import os
import time
from playwright.sync_api import sync_playwright

ASSETS_DIR = r"f:\Projects !!\debrisscanAI - new\space-debris-ai\assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

def run():
    with sync_playwright() as p:
        print("[*] Launching Chromium instance...")
        # Headless mode can sometimes clip WebGL; launching with args to ensure good rendering.
        browser = p.chromium.launch(headless=True, args=['--use-angle=default'])
        # Streamlit dashboards look better wide.
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        
        print(f"[*] Navigating to Streamlit Dashboard...")
        page.goto("http://127.0.0.1:8501")
        
        # Wait for the system to boot and the success message to be present.
        print("[*] Waiting for application load...")
        page.wait_for_selector("text=System Online", timeout=60000)
        
        print("[*] Waiting for Plotly WebGL to render...")
        time.sleep(15) 
        
        print("[*] Capturing Tab 1 (Simulation)...")
        page.screenshot(path=os.path.join(ASSETS_DIR, "simulation_tab.png"), full_page=True)
        
        print("[*] Switching to Tab 2 (Metrics)...")
        page.locator("button", has_text="Research Metrics & Evaluation").click()
        time.sleep(5)
        print("[*] Capturing Tab 2...")
        page.screenshot(path=os.path.join(ASSETS_DIR, "metrics_tab.png"), full_page=True)
        
        print("[*] Switching to Tab 3 (Pipeline)...")
        page.locator("button", has_text="True E2E Inference").click()
        time.sleep(10) # Heavy image loading
        print("[*] Capturing Tab 3...")
        page.screenshot(path=os.path.join(ASSETS_DIR, "pipeline_tab.png"), full_page=True)
        
        print("[*] Switching to Tab 4 (Removal)...")
        page.locator("button", has_text="Autonomous Removal Sim").click()
        time.sleep(8)
        
        print("[*] Injecting AI Rocket Interaction...")
        try:
            page.locator("button", has_text="Launch AI Interceptor").click()
            time.sleep(6) # Give the rocket time to travel bounds
        except:
            pass
            
        print("[*] Capturing Tab 4...")
        page.screenshot(path=os.path.join(ASSETS_DIR, "removal_tab.png"), full_page=True)
        
        browser.close()
        print("[✔] All high-resolution screenshots generated successfully in assets/ directory.")

if __name__ == "__main__":
    run()
