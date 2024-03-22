def show_title(title):
    title_length = len(title)
    width = title_length + 2

    print(" ","-" * width)
    print(f"| {title.center(title_length + 2)} |")
    print(" ","-" * width)
