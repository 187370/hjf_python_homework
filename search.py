import requests
from typing import Optional

SEARXNG_URL = "https://searxng.tblu.xyz"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AI-Research-Bot)"
}

def search_searxng(
    query: str,
    current_date: Optional[str] = None,
    time_range: int = 30,
    category: str = "general",
    lang: str = "zh-CN",
    page: int = 1,
    engines: Optional[str] = None,
    return_results: bool = False,
):
    """使用 searxng 搜索指定内容并返回结果列表"""

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

    if not return_results:
        print(f"\n 正在搜索：{full_query}\n")

    response = requests.get(f"{SEARXNG_URL}/search", params=params, headers=HEADERS)

    if response.status_code != 200:
        if not return_results:
            print(" 搜索失败:", response.status_code, response.text)
        return []

    results = response.json().get("results", [])
    if not results:
        if not return_results:
            print(" 没有找到结果")
        return []

    formatted_results = []
    for r in results:
        formatted_results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
        )

    if not return_results:
        for i, r in enumerate(formatted_results, 1):
            print(f"{i}. {r['title']}\n    摘要：{r['content']}\n    链接：{r['url']}\n")

    return formatted_results


if __name__ == "__main__":
    formatted_results=search_searxng(
        "AAPL股票？",
        current_date="2025-6-10",
        time_range=30,
        category="news",
        engines="google,bing"
    )
    print(formatted_results)
