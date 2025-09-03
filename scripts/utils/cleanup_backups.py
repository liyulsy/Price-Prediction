#!/usr/bin/env python3
"""
清理重复的备份文件脚本
"""

import os
import glob
from datetime import datetime

def cleanup_backup_files():
    """清理重复的备份文件，只保留最新的一个"""
    
    # 查找所有备份文件
    backup_patterns = [
        'scripts/training/*.backup*',
        '*.backup*',
        'scripts/experiments/*.backup*'
    ]
    
    all_backups = []
    for pattern in backup_patterns:
        all_backups.extend(glob.glob(pattern))
    
    if not all_backups:
        print("✅ 没有找到备份文件")
        return
    
    print(f"🔍 找到 {len(all_backups)} 个备份文件:")
    for backup in all_backups:
        size = os.path.getsize(backup) / 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(backup))
        print(f"   {backup} ({size:.1f}KB, {mtime.strftime('%Y-%m-%d %H:%M')})")
    
    # 按原始文件分组
    backup_groups = {}
    for backup in all_backups:
        # 提取原始文件名
        if '.backup' in backup:
            original = backup.split('.backup')[0]
            if original not in backup_groups:
                backup_groups[original] = []
            backup_groups[original].append(backup)
    
    # 对每组备份文件，只保留最新的
    total_removed = 0
    total_saved_space = 0
    
    for original, backups in backup_groups.items():
        if len(backups) <= 1:
            continue
            
        # 按修改时间排序，保留最新的
        backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest = backups[0]
        to_remove = backups[1:]
        
        print(f"\n📁 原始文件: {original}")
        print(f"   ✅ 保留: {latest}")
        
        for backup in to_remove:
            size = os.path.getsize(backup)
            total_saved_space += size
            total_removed += 1
            print(f"   🗑️ 删除: {backup}")
            os.remove(backup)
    
    if total_removed > 0:
        print(f"\n🎉 清理完成!")
        print(f"   删除了 {total_removed} 个重复备份文件")
        print(f"   节省空间: {total_saved_space/1024:.1f} KB")
    else:
        print("\n✅ 没有重复的备份文件需要清理")

def main():
    """主函数"""
    print("🧹 开始清理重复的备份文件...")
    cleanup_backup_files()

if __name__ == "__main__":
    main()
