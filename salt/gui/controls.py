from multiprocessing import Queue
from time import time, localtime, strftime
from Queue import Empty as EmptyQueue
from six.moves import tkinter as tk
from six import PY3
if PY3:
    from tkinter import ttk
else:
    import ttk


# DEPRECATED
class Console(ttk.Tkinter.Text):
    def __init__(self, master, **options):
        ttk.Tkinter.Text.__init__(self, master, **options)
        self.queue = Queue()  # queue
        self.update_content()

    def write(self, line):
        self.queue.put(line)

    def clear(self):
        self.queue.put(None)

    def update_content(self):
        try:
            while True:
                line = self.queue.get_nowait()
                if line is None:
                    self.config(state=tk.NORMAL)
                    self.delete(1.0, ttk.Tkinter.END)
                else:
                    self.config(state=tk.NORMAL)
                    self.insert(ttk.Tkinter.END, line)
                self.see(ttk.Tkinter.END)
                self.update_idletasks()
                #self.config(state=tk.DISABLED)
        except EmptyQueue:
            pass
        self.after(100, self.update_content)


"""
Following source code (ToolTip class) taken from
http://code.activestate.com/recipes/576688-tooltip-for-tkinter/
"""

class ToolTip( tk.Toplevel ):
    """
    Provides a ToolTip widget for Tkinter.
    To apply a ToolTip to any Tkinter widget, simply pass the widget to the
    ToolTip constructor
    """
    def __init__( self, wdgt, msg=None, msgFunc=None, delay=0.7, follow=True ):
        """
        Initialize the ToolTip

        Arguments:
          wdgt: The widget this ToolTip is assigned to
          msg:  A static string message assigned to the ToolTip
          msgFunc: A function that retrieves a string to use as the ToolTip text
          delay:   The delay in seconds before the ToolTip appears(may be float)
          follow:  If True, the ToolTip follows motion, otherwise hides
        """
        self.wdgt = wdgt
        self.parent = self.wdgt.master                                          # The parent of the ToolTip is the parent of the ToolTips widget
        tk.Toplevel.__init__( self, self.parent, bg='black', padx=1, pady=1 )      # Initalise the Toplevel
        self.withdraw()                                                         # Hide initially
        self.overrideredirect( True )                                           # The ToolTip Toplevel should have no frame or title bar

        self.msgVar = tk.StringVar()                                               # The msgVar will contain the text displayed by the ToolTip
        if msg == None:
            self.msgVar.set( 'No message provided' )
        else:
            self.msgVar.set( msg )
        self.msgFunc = msgFunc
        self.delay = delay
        self.follow = follow
        self.visible = 0
        self.lastMotion = 0
        tk.Message( self, textvariable=self.msgVar, bg='#FFFFDD',
                 aspect=1000 ).grid()                                           # The test of the ToolTip is displayed in a Message widget
        self.wdgt.bind( '<Enter>', self.spawn, '+' )                            # Add bindings to the widget.  This will NOT override bindings that the widget already has
        self.wdgt.bind( '<Leave>', self.hide, '+' )
        self.wdgt.bind( '<Motion>', self.move, '+' )

    def spawn( self, event=None ):
        """
        Spawn the ToolTip.  This simply makes the ToolTip eligible for display.
        Usually this is caused by entering the widget

        Arguments:
          event: The event that called this funciton
        """
        self.visible = 1
        self.after( int( self.delay * 1000 ), self.show )                       # The after function takes a time argument in miliseconds

    def show( self ):
        """
        Displays the ToolTip if the time delay has been long enough
        """
        if self.visible == 1 and time() - self.lastMotion > self.delay:
            self.visible = 2
        if self.visible == 2:
            self.deiconify()

    def move( self, event ):
        """
        Processes motion within the widget.

        Arguments:
          event: The event that called this function
        """
        self.lastMotion = time()
        if self.follow == False:                                                # If the follow flag is not set, motion within the widget will make the ToolTip dissapear
            self.withdraw()
            self.visible = 1
        self.geometry( '+%i+%i' % ( event.x_root+10, event.y_root+10 ) )        # Offset the ToolTip 10x10 pixes southwest of the pointer
        try:
            self.msgVar.set( self.msgFunc() )                                   # Try to call the message function.  Will not change the message if the message function is None or the message function fails
        except:
            pass

        self.after( int( self.delay * 1000 ), self.show )

    def hide( self, event=None ):
        """
        Hides the ToolTip.  Usually this is caused by leaving the widget

        Arguments:
          event: The event that called this function
        """
        self.visible = 0
        self.withdraw()
