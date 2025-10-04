#!/usr/bin/env python3
"""
ALM Project Setup Script
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = ['checkpoints', 'logs', 'results']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/")
    return True

def check_dataset():
    """Check if dataset is available."""
    print("🔍 Checking dataset...")
    if Path("master_metadata.csv").exists():
        print("   ✅ Dataset metadata found")
        return True
    else:
        print("   ⚠️  Dataset metadata not found")
        print("   💡 Please ensure you have the dataset in the correct location")
        return False

def main():
    """Main setup function."""
    print("🚀 ALM PROJECT SETUP")
    print("="*50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed at directory creation")
        return False
    
    # Check dataset
    dataset_available = check_dataset()
    
    print("\n🎉 SETUP COMPLETE!")
    print("="*30)
    print("📋 Next steps:")
    print("   1. Train models: python train_alm.py")
    print("   2. Test on root files: python test_root_files.py")
    print("   3. Test on dataset: python test_dataset.py")
    
    if not dataset_available:
        print("\n⚠️  Note: Dataset not found. Please add your dataset before training.")
    
    return True

if __name__ == "__main__":
    main()
