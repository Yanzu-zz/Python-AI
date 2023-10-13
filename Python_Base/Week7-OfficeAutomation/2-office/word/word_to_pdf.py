import pdfkit

# pdfkit.from_url('https://www.baidu.com', './text1.pdf')
html = """
<html>
<head>
<meta charset="utf-8" />
</head>
<body>
    <p>this is test pdf generation</p>
</body>
</html>
"""
pdfkit.from_string(html, 'test1.pdf')
