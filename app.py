import flet as ft

def main(page: ft.Page):
    page.title = "My Python App"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Create a text element to show the count
    counter = ft.Text("0", size=50, weight=ft.FontWeight.BOLD)

    # Function to increase the count
    def increment(e):
        counter.value = str(int(counter.value) + 1)
        page.update()

    # Add elements to the page
    page.add(
        ft.Column(
            [
                ft.Text("Hello from Python!", size=24, color="blue"),
                ft.Container(content=counter, margin=10),
                ft.ElevatedButton("Click Me", on_click=increment, width=200),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
