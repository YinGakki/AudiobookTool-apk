import re
import sys
from pathlib import Path

def update_version(new_version):
    # 更新 Android 版本
    build_gradle = Path('android/app/build.gradle')
    content = build_gradle.read_text()
    
    # 提取版本号并递增
    version_match = re.search(r'versionCode (\d+)', content)
    if version_match:
        version_code = int(version_match.group(1)) + 1
        content = re.sub(r'versionCode \d+', f'versionCode {version_code}', content)
    
    content = re.sub(r'versionName "[^"]*"', f'versionName "{new_version}"', content)
    build_gradle.write_text(content)
    
    print(f"Updated Android version to {new_version} (code: {version_code})")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    update_version(sys.argv[1])
