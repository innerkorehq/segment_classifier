from segment_classifier.utils.html_normalizer import normalize_segment

html = '<div class="card structural"><h1 class="title">Hello</h1><p role="description">World</p></div>'
text = "Hello World"
normalized = normalize_segment(html, text)

print(f"Skeleton: {normalized.skeleton}")
print(f"Normalized HTML: {normalized.normalized_html}")
