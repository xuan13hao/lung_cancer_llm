import sys
import requests
from bs4 import BeautifulSoup
import re
import os

def extract_urls_from_obo(file_path):
    """
    从 OBO 文件中提取所有 `def` 字段中的 URL，去掉前缀 'https://'。
    """
    urls = []
    with open(file_path, 'r', encoding='utf-8') as obo_file:
        for line in obo_file:
            # 匹配 def 行中的 URL
            if line.startswith("def:"):
                found_urls = re.findall(r'url:https\\://(\S+)', line)
                urls.extend(found_urls)
    return urls

def fetch_web_content(url):
    """
    爬取网页并提取主要文本内容。
    """
    try:
        full_url = f"https://{url}"
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # 提取正文内容
        text = soup.get_text(separator='\n', strip=True)
        return text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}", file=sys.stderr)
        return None

def save_content_to_file(content, file_name):
    """
    将爬取的内容保存到文件。
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """
    主程序，通过命令行输入控制文件路径和输出目录。
    """
    if len(sys.argv) != 3:
        print("Usage: python script.py <obo_file_path> <output_directory>", file=sys.stderr)
        sys.exit(1)
    
    obo_file_path = sys.argv[1]
    output_dir = sys.argv[2]

    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 提取 URL
    urls = extract_urls_from_obo(obo_file_path)
    print(f"Found {len(urls)} URLs to process.")
    
    for idx, url in enumerate(urls, start=1):
        print(f"Fetching content from URL {idx}/{len(urls)}: https://{url}")
        content = fetch_web_content(url)
        if content:
            output_file = os.path.join(output_dir, f"page_{idx}.txt")
            save_content_to_file(content, output_file)
            print(f"Saved content to {output_file}.")
        else:
            print(f"Skipping URL {url} due to fetch failure.", file=sys.stderr)

if __name__ == "__main__":
    main()
