def display_progress_bar(progress, total_size):
    """Função responsável por exibir a barra de progresso."""
    done = int(50 * progress / total_size)
    print(f"\r[{'=' * done}{' ' * (50-done)}] {progress / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB", end='')
    