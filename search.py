import requests
from typing import Optional

SEARXNG_URL = "https://searxng.tblu.xyz"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AI-Research-Bot)"
}

def search_searxng(
    query: str,
   
    current_date: Optional[str] = None,
    time_range: int=30,
    category: str = "general",
    lang: str = "zh-CN",
    page: int = 1,
    engines: Optional[str] = None,
):
    # 构造时间语法
    date_filter = f"查询{current_date}前{time_range}之内"
    
    full_query = f"{date_filter}{query}"

    params = {
        "q": full_query,
        "format": "json",
        "lang": lang,
        "categories": category,
        "page": page,
    }

    if engines:
        params["engines"] = engines

    print(f"\n 正在搜索：{full_query}\n")
    response = requests.get(f"{SEARXNG_URL}/search", params=params, headers=HEADERS)
    
    if response.status_code != 200:
        print(" 搜索失败:", response.status_code, response.text)
        return

    results = response.json().get("results", [])
    if not results:
        print(" 没有找到结果")
        return

    for i, r in enumerate(results, 1):
        title = r.get("title", "（无标题）")
        url = r.get("url", "")
        content = r.get("content", "（无摘要）")

        print(f"{i}. {title}\n    摘要：{content}\n    链接：{url}\n")


if __name__ == "__main__":
    search_searxng(
        "AAPL股票值得买吗？",
        current_date="2012-2-28",
        time_range=30,
        category="news",
        engines="google,bing"
    )
