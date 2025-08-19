import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def compute_output_size(input_size, kernel_size, stride, padding):
    """(input_size + 2*padding - kernel_size) // stride + 1"""
    return (input_size + 2*padding - kernel_size) // stride + 1

def display_formula(input_size, kernel_size, stride, padding):
    out_size = compute_output_size(input_size, kernel_size, stride, padding)
    txt = (
        "Output Size = \n"
        " [(input_size + 2*padding - kernel_size) / stride] + 1 = \n"
    )
    txt2 = f" [({input_size} + 2*{padding} - {kernel_size}) / {stride}] + 1 = {out_size}"
    return txt + txt2

def draw_grid(ax, input_size, kernel_size, stride, padding, kernel_pos):
    """
    1) Label zone at y in [0, 2].
    2) Grid zone at [2, 2 + global_height].
    3) Input top row near grid_top-1, output top row near grid_top-1.
    4) Negative i => higher in the figure. 
    """
    ax.clear()

    padded_size = input_size + 2*padding
    output_size = compute_output_size(input_size, kernel_size, stride, padding)

    # We'll define global_height so we have enough vertical space for whichever is bigger.
    global_height = max(padded_size, output_size)

    # The grids occupy [grid_bottom, grid_top].
    grid_bottom = 2
    grid_top    = 2 + global_height

    # We'll define input_vertical_shift so that the top row of input (i=-padding) is at y= grid_top-1.
    input_vertical_shift = (grid_top - 1) - padding
    # We'll define output_vertical_shift so that out_i=0 is at y= grid_top-1.
    output_vertical_shift = (grid_top - 1)

    # We'll place the output to the right of the input
    x_output_offset = padded_size + 3

    x_start, y_start = kernel_pos

    # ---- 1) Draw the padded input grid ----
    # i in [-padding, ..., input_size+padding-1]
    # y = -i + input_vertical_shift
    for i in range(-padding, input_size+padding):
        for j in range(-padding, input_size+padding):
            if (i<0 or i>=input_size or j<0 or j>=input_size):
                color = 'lightgray'
            else:
                color = 'white'
            y_coord = -i + input_vertical_shift
            ax.add_patch(
                plt.Rectangle((j, y_coord), 1, 1, facecolor=color, edgecolor='black')
            )

    # ---- 2) Draw the output grid ----
    for out_i in range(output_size):
        for out_j in range(output_size):
            y_coord = -out_i + output_vertical_shift
            x_coord = x_output_offset + out_j
            ax.add_patch(
                plt.Rectangle((x_coord, y_coord), 1, 1, fill=False, edgecolor='green')
            )

    # ---- 3) Highlight the red kernel in input ----
    x0, y0 = kernel_pos
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            xx = x0 + dx
            yy = y0 + dy
            # If inside padded region, color it
            if -padding <= xx < (input_size+padding) and -padding <= yy < (input_size+padding):
                y_coord = -yy + input_vertical_shift
                ax.add_patch(
                    plt.Rectangle((xx, y_coord), 1, 1, facecolor='red', alpha=0.5)
                )

    # ---- 4) Highlight the corresponding output pixel (orange) ----
    out_x = (x0 + padding)//stride
    out_y = (y0 + padding)//stride
    if 0 <= out_x < output_size and 0 <= out_y < output_size:
        x_coord = x_output_offset + out_x
        y_coord = -out_y + output_vertical_shift
        ax.add_patch(
            plt.Rectangle((x_coord, y_coord), 1, 1, facecolor='orange', alpha=0.5)
        )

    # ---- 5) Axis limits ----
    x_min = -padding - 2
    x_max = x_output_offset + output_size + 2
    y_min = 0  # label zone bottom
    y_max = grid_top + 1  # a bit more space above the grid
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.axis('off')

    # ---- 6) Labels "Input" & "Output" near y=1 ----
    label_y = 1
    input_center_x = 0.5 * ((-padding) + (input_size + padding - 1))
    output_center_x = x_output_offset + 0.5*(output_size-1)

    ax.text(
        input_center_x, label_y,
        "Input",
        ha='center', va='top', fontsize=12
    )
    ax.text(
        output_center_x, label_y,
        "Output",
        ha='center', va='top', fontsize=12
    )

def create_widget():
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.close(fig)

    input_size_slider = widgets.IntSlider(value=5, min=3, max=20, step=1, description='Input Size:')
    kernel_size_slider = widgets.IntSlider(value=3, min=1, max=19, step=2, description='Kernel Size:')
    stride_slider      = widgets.IntSlider(value=1, min=1, max=5, step=1, description='Stride:')
    padding_slider     = widgets.IntSlider(value=0, min=0, max=10, step=1, description='Padding:')
    x_pos_slider       = widgets.IntSlider(value=0, min=-5, max=15, step=1, description='Kernel X:')
    y_pos_slider       = widgets.IntSlider(value=0, min=-5, max=15, step=1, description='Kernel Y:')

    plot_output = widgets.Output()
    with plot_output:
        display(fig)

    formula_output = widgets.Textarea(
        value="", description="Formula:",
        layout=widgets.Layout(width='60%', height='60px')
    )

    def update_plot(change=None):
        input_size  = input_size_slider.value
        kernel_size = kernel_size_slider.value
        stride      = stride_slider.value
        padding     = padding_slider.value

        out_size = compute_output_size(input_size, kernel_size, stride, padding)
        x_min = -padding
        x_max = (out_size - 1)*stride - padding if out_size > 0 else -padding
        if x_min > x_max:
            x_min = x_max
        x_pos_slider.min = x_min
        x_pos_slider.max = x_max

        y_min = -padding
        y_max = (out_size - 1)*stride - padding if out_size > 0 else -padding
        if y_min > y_max:
            y_min = y_max
        y_pos_slider.min = y_min
        y_pos_slider.max = y_max

        x_pos_slider.step = stride
        y_pos_slider.step = stride

        x_start = x_pos_slider.value
        y_start = y_pos_slider.value

        # Update formula
        formula_output.value = display_formula(input_size, kernel_size, stride, padding)

        draw_grid(ax, input_size, kernel_size, stride, padding, (x_start, y_start))

        with plot_output:
            plot_output.clear_output(wait=True)
            display(fig)
            fig.canvas.draw()

    def reset_kernel(change=None):
        x_pos_slider.value = 0
        y_pos_slider.value = 0
        update_plot()

    for s in [input_size_slider, kernel_size_slider, stride_slider, padding_slider]:
        s.observe(reset_kernel, 'value')

    x_pos_slider.observe(update_plot, 'value')
    y_pos_slider.observe(update_plot, 'value')

    controls = widgets.VBox([
        input_size_slider, kernel_size_slider, stride_slider,
        padding_slider, x_pos_slider, y_pos_slider
    ], layout=widgets.Layout(width='40%'))

    display(widgets.HBox([controls, formula_output]))
    display(plot_output)

    update_plot()