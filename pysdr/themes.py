from bokeh.themes import Theme
import yaml

# nice way of providing a theme for all the figures in the Bokeh document
# note that this is only for Bokeh stuff, the HTML page has it's own css as well
    
black_and_white = Theme(json=yaml.load("""
attrs:
    Figure:
        background_fill_color: "#333333"
        outline_line_color: white
        toolbar_location: right
        border_fill_color: black
    Axis:
        axis_label_text_color: "white"
        major_label_text_color: "white"
        axis_label_text_font_style: bold
        major_tick_line_color: "white"
        minor_tick_line_color: "white"
    Title:
        text_color: "white"
        text_font_style: bold
    Grid:
        grid_line_dash: [2, 2]
        grid_line_color: gray 
"""))
