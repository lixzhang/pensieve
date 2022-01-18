# https://www.geeksforgeeks.org/create-a-watchdog-in-python-to-look-for-filesystem-changes/

# import time module, Observer, FileSystemEventHandler
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, DirCreatedEvent
import os
import sys

class OnMyWatch:
    # Set the directory on watch
    watchDirectory = "/home/lixun/Desktop/pensieve/sim/"
  
    def __init__(self):
        self.observer = Observer()
  
    def run(self):
        event_handler = HandlerDir() #  Handler()
        self.observer.schedule(event_handler, self.watchDirectory, recursive = True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            print("Observer Stopped")
  
        self.observer.join()
  
  
class Handler(FileSystemEventHandler):
  
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            print("directory created")

        # if event.event_type == 'created':
        #     # Event is created, you can process it now
        #     print("Watchdog received created event - % s." % event.src_path)
        # elif event.event_type == 'modified':
        #     # Event is modified, you can process it now
        #     print("Watchdog received modified event - % s." % event.src_path)

class HandlerDir(FileSystemEventHandler):
    counter = int(sys.argv[1])
    def on_created(self, event):
        if event.is_directory:
            print(event.src_path, event.event_type)
            time.sleep(20)
            os.system("python plot_results.py " + str(self.counter))
            print("Counter", self.counter)
            print("done")            
            self.counter += 100

    # @staticmethod
    # def on_any_event(event):
    #     time.sleep(3)
    #     print("folder created")

if __name__ == '__main__':
    watch = OnMyWatch()
    watch.run()
