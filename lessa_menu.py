"""
LESSA Main Menu - Central hub for all LESSA tools and demos
Provides organized access to different components of the sign language system.
"""

import sys
import os
from typing import Optional

def print_lessa_header():
    """Print LESSA header with El Salvador flag colors."""
    print("=" * 70)
    print("🇸🇻 LESSA - El Salvador Sign Language System")
    print("=" * 70)
    print("Advanced computer vision system for sign language recognition")
    print("Powered by MediaPipe Holistic Detection")
    print()

def print_main_menu():
    """Display the main menu options."""
    print("📋 LESSA Main Menu:")
    print("-" * 30)
    print("1. 🎯 Basic LESSA Demo")
    print("   └─ Simple hand detection and pose tracking")
    print()
    print("2. 🚀 Enhanced LESSA Demo") 
    print("   └─ Advanced camera management & quality assessment")
    print()
    print("3. 🔤 Alphabet Collection Tool")
    print("   └─ Systematic A-Z letter data collection")
    print()
    print("4. 📊 System Information")
    print("   └─ Camera detection & system diagnostics")
    print()
    print("5. 📚 Documentation")
    print("   └─ View guides and documentation")
    print()
    print("0. ❌ Exit")
    print()

def run_basic_demo():
    """Run the basic LESSA demo."""
    print("🎯 Starting Basic LESSA Demo...")
    print("Loading holistic detection system...")
    
    try:
        # Import and run the basic demo
        from lessa_demo import main
        main()
    except ImportError as e:
        print(f"❌ Error importing basic demo: {e}")
        print("Make sure lessa_demo.py exists in the project directory")
    except Exception as e:
        print(f"❌ Error running basic demo: {e}")

def run_enhanced_demo():
    """Run the enhanced LESSA demo with camera management."""
    print("🚀 Starting Enhanced LESSA Demo...")
    print("Initializing advanced camera management...")
    
    try:
        # Import and run the enhanced demo
        from lessa_enhanced_demo import main
        main()
    except ImportError as e:
        print(f"❌ Error importing enhanced demo: {e}")
        print("Make sure lessa_enhanced_demo.py exists in the project directory")
    except Exception as e:
        print(f"❌ Error running enhanced demo: {e}")

def run_alphabet_collector():
    """Run the alphabet collection tool."""
    print("🔤 Starting Alphabet Collection Tool...")
    print("Preparing systematic letter data collection...")
    
    try:
        # Import and run the alphabet collector
        from alphabet_collector import run_alphabet_collector
        run_alphabet_collector()
    except ImportError as e:
        print(f"❌ Error importing alphabet collector: {e}")
        print("Make sure alphabet_collector.py exists in the project directory")
    except Exception as e:
        print(f"❌ Error running alphabet collector: {e}")

def show_system_info():
    """Display system information and diagnostics."""
    print("📊 LESSA System Information")
    print("=" * 40)
    
    try:
        # Add the src directory to the path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.utils.enhanced_camera import CameraManager
        
        # Detect cameras
        print("🔍 Detecting cameras...")
        cameras = CameraManager.detect_cameras()
        
        if cameras:
            print(CameraManager.format_camera_list(cameras))
            
            # Show best camera recommendation
            best_camera = CameraManager.get_best_camera()
            if best_camera:
                print(f"💡 Recommended camera: Camera {best_camera.device_id}")
                print(f"   Quality Score: {best_camera.quality_score:.1f}/100")
        else:
            print("❌ No cameras detected")
        
        # System requirements check
        print("\n🔧 System Requirements:")
        print("✅ Python version:", sys.version.split()[0])
        
        # Check key dependencies
        dependencies = [
            ("opencv-python", "cv2"),
            ("mediapipe", "mediapipe"),
            ("numpy", "numpy")
        ]
        
        for package_name, import_name in dependencies:
            try:
                __import__(import_name)
                print(f"✅ {package_name}: Installed")
            except ImportError:
                print(f"❌ {package_name}: Missing - run 'pip install {package_name}'")
        
    except Exception as e:
        print(f"❌ Error getting system information: {e}")
    
    input("\nPress Enter to return to main menu...")

def show_documentation():
    """Show documentation menu."""
    print("📚 LESSA Documentation")
    print("=" * 30)
    print("1. 📖 System Overview (LESSA_OVERVIEW.md)")
    print("2. 📹 Camera System Guide (CAMERA_SYSTEM_DOCS.md)")
    print("3. 🔧 Technical Requirements")
    print("4. 🎯 Usage Examples")
    print("0. ← Back to Main Menu")
    print()
    
    while True:
        try:
            choice = input("Select documentation (0-4): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                show_file_content("LESSA_OVERVIEW.md", "System Overview")
            elif choice == '2':
                show_file_content("CAMERA_SYSTEM_DOCS.md", "Camera System Guide") 
            elif choice == '3':
                show_technical_requirements()
            elif choice == '4':
                show_usage_examples()
            else:
                print("❌ Invalid choice. Please select 0-4.")
        
        except (KeyboardInterrupt, EOFError):
            break

def show_file_content(filename: str, title: str):
    """Display content of a documentation file."""
    try:
        if os.path.exists(filename):
            print(f"\n📖 {title}")
            print("=" * (len(title) + 4))
            print(f"File: {filename}")
            print("-" * 50)
            print("Opening in default text viewer...")
            
            # Try to open with default system viewer
            if sys.platform == "win32":
                os.system(f'start notepad "{filename}"')
            elif sys.platform == "darwin":
                os.system(f'open "{filename}"')
            else:
                os.system(f'xdg-open "{filename}"')
            
        else:
            print(f"❌ Documentation file '{filename}' not found")
            print("Make sure you're running from the project root directory")
    
    except Exception as e:
        print(f"❌ Error opening documentation: {e}")
    
    input("\nPress Enter to continue...")

def show_technical_requirements():
    """Show technical requirements."""
    print("\n🔧 LESSA Technical Requirements")
    print("=" * 40)
    
    requirements = [
        ("Python Version", "3.8 - 3.11 (MediaPipe compatibility)"),
        ("Operating System", "Windows 10/11, macOS, Linux"),
        ("RAM", "8GB minimum, 16GB recommended"),
        ("Camera", "USB webcam or built-in camera"),
        ("USB", "USB 2.0 minimum, USB 3.0 recommended"),
        ("Processor", "Multi-core CPU for real-time processing")
    ]
    
    for component, requirement in requirements:
        print(f"• {component}: {requirement}")
    
    print("\n📦 Required Python Packages:")
    packages = [
        "opencv-python>=4.5.0",
        "mediapipe>=0.10.0", 
        "numpy>=1.21.0",
        "streamlit>=1.25.0 (optional, for web interface)"
    ]
    
    for package in packages:
        print(f"• {package}")
    
    print("\nInstall all packages with:")
    print("pip install -r requirements.txt")
    
    input("\nPress Enter to continue...")

def show_usage_examples():
    """Show usage examples."""
    print("\n🎯 LESSA Usage Examples")
    print("=" * 30)
    
    examples = [
        ("Basic Hand Detection", "python lessa_demo.py"),
        ("Enhanced Camera Demo", "python lessa_enhanced_demo.py"),
        ("Alphabet Collection", "python alphabet_collector.py"),
        ("Main Menu", "python lessa_menu.py")
    ]
    
    print("Command Line Usage:")
    for description, command in examples:
        print(f"• {description}:")
        print(f"  {command}")
        print()
    
    print("Interactive Controls:")
    print("• Q - Quit any demo")
    print("• F - Toggle feature display")
    print("• H - Toggle help overlay")
    print("• S - Save detection sample")
    print("• C - Toggle camera quality (enhanced demo)")
    print("• SPACE - Save alphabet sample (alphabet collector)")
    print("• N/P - Next/Previous letter (alphabet collector)")
    
    input("\nPress Enter to continue...")

def get_user_choice() -> Optional[str]:
    """Get user menu choice with error handling."""
    try:
        return input("Select an option (0-5): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n👋 Goodbye!")
        return '0'

def main():
    """Main menu loop."""
    print_lessa_header()
    
    while True:
        print_main_menu()
        choice = get_user_choice()
        
        if choice == '0':
            print("👋 Thank you for using LESSA!")
            print("🇸🇻 ¡Hasta luego!")
            break
        elif choice == '1':
            run_basic_demo()
        elif choice == '2':
            run_enhanced_demo()
        elif choice == '3':
            run_alphabet_collector()
        elif choice == '4':
            show_system_info()
        elif choice == '5':
            show_documentation()
        else:
            print("❌ Invalid choice. Please select 0-5.")
        
        print("\n" + "=" * 70)

if __name__ == "__main__":
    main()