# plagiarism-detector

pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown animal leaps over a sleeping dog.",
    "The cat sat on the mat and looked around.",
    "Dogs are usually lazy, especially when the weather is warm.",
    "A brown fox swiftly jumps over dogs sleeping in the sun.",
    "The fox, brown and fast, leaped over the lazy animal.",
]


texts = [text.lower().strip() for text in texts]


for i, text in enumerate(texts):
    print(f"Text {i+1} (Preprocessed): {text}")


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

print("\nTF-IDF Words:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix (Numbers):")
print(tfidf_matrix.toarray())


print("\n Pair-wise Cosine Similarities:\n")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        similarity = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0]
        print(f"Text {i+1} vs Text {j+1} => Similarity: {similarity:.2f}")
        if similarity > 0.75:
            print("High similarity! Possible Plagiarism.")
        elif similarity > 0.5:
            print("Moderate similarity. Needs review.")
        else:
            print("  Low similarity. Looks original.")
