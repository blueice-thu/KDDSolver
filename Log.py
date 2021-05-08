from datetime import datetime


class Log:
    def __init__(self, path="logs", show=True):
        time_str = "log_" + str(datetime.now()).split(".")[0].replace("-", "_").replace(":", "_").replace(" ", "_")
        self.log = open(path + "/" + time_str + ".txt", 'w')
        self.show = show

    def write(self, content: str, end="\n"):
        if self.show:
            print(content, end=end)
        self.log.write(content + end)

    def __del__(self):
        self.log.close()
