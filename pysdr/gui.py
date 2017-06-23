from bokeh.plotting import figure
from bokeh.models import WheelZoomTool, BoxZoomTool, ResetTool, SaveTool # all the tools we want- reference http://bokeh.pydata.org/en/0.10.0/docs/reference/models/tools.html
from bokeh.models import Range1d

'''
Wrap Bokeh's main figure object. 
   The point of wrapping it is to "hardcode" the stuff like ratio, tools, hiding logo, etc
   so as to reduce the amount of code in the main script
   it also lets us add more intuitive functions, like _set_x_range as the perfect example
   the goal of this wrapper is to provide stuff a lot of people will want to use
   remember, the user can choose to use Bokeh directly, so lets not go crazy with features
'''
def base_plot(x_label, y_label, title, **kwargs):
    
    # This allows disabling of horizontal zooming, which gets annoying in most dsp plots
    # e.g. usage: fft_plot = pysdr.base_plot('Freq', 'PSD', 'Frequency', disable_horizontal_zooming=True)
    if 'disable_horizontal_zooming' in kwargs and kwargs['disable_horizontal_zooming']: # if it's specified and is set True
        tools = [WheelZoomTool(dimensions='height')]
    else:
        tools = [WheelZoomTool()]

    # Similar to above, except disable all zooming, perfect for waterfall plots
    if 'disable_all_zooming' in kwargs and kwargs['disable_all_zooming']: # if it's specified and is set True
        tools = [] # removes the WheelZoomTool we just added above
            
    # Create the Bokeh figure
    plot = figure(plot_width = 300, # this is more for the ratio, because we have auto-width scaling
                  plot_height = 200,
                  y_axis_label = y_label,
                  x_axis_label = x_label,
                  tools = tools + [BoxZoomTool(), ResetTool(), SaveTool()], # all the other tools we want- reference http://bokeh.pydata.org/en/0.10.0/docs/reference/models/tools.html
                  title = title)  # use min_border=30 to add padding between plots, if we ever want it
    
    # sets wheel zoom active by default (tools[0] is the wheelzoom), unless zooming was disabled
    if 'disable_all_zooming' not in kwargs:
        plot.toolbar.active_scroll = plot.toolbar.tools[0] 
    
    # hides stupid bokeh logo
    plot.toolbar.logo = None 
    
    # add more intuitive functions to set x and y ranges
    def _set_x_range(min_x, max_x): # without the underscore it wont work, bokeh/core/has_props.py overloads __setattr__ to intercept attribute setting that is not private
        plot.x_range = Range1d(min_x, max_x)
    def _set_y_range(min_y, max_y):
        plot.y_range = Range1d(min_y, max_y)
    plot._set_x_range = _set_x_range # add functions to object
    plot._set_y_range = _set_y_range
    
    # return the bokeh figure object
    return plot 
    

# The idea behind this utilization bar is to have an "included by default" widget to show
#    how well the process_samples is keeping up with the incoming samples, in a realtime manner
def utilization_bar():
    plot = figure(plot_width = 150, # this is more for the ratio, because we have auto-width scaling
                  plot_height = 150,
                  tools = [], # no tools needed for this one
                  title = 'Utilization')
    plot.toolbar.logo = None  # hides logo
    plot.x_range = Range1d(0, 1) 
    plot.y_range = Range1d(0, 0.1)  # using 0.1 instead of 1 just so we can see it move
    plot.xaxis.visible = False # hide x axis
    return plot



