from langchain_community.retrievers import ArxivRetriever

retrievers = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

docs = retrievers.invoke("Large Language Model")

for i, doc in enumerate(docs):
    print(f"\nResult {1+1}")
    print("Title : ", doc.metadata.get("Title"))
    print("Authors : ", doc.metadata.get("Authors"))
    print("Summary : ", doc.page_content[:500])