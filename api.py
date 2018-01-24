import zerorpc

import os
import sys
sys.path.insert(0, '/Users/jlaura/github/autocnet')

from autocnet.graph.network import CandidateGraph

def parse_port():
    return 4242

class AutocnetAPI(object):

    def echo(self, text):
        return text

    def new(self):
        self.cg = CandidateGraph()
        return 'success'

    def about(self):
        return len(self.cg)

    def open(self, fname):
        return fname

def main():
    addr = 'tcp://127.0.0.1:{}'.format(parse_port())
    s = zerorpc.Server(AutocnetAPI())
    s.bind(addr)
    s.run()

if __name__ == '__main__':
    main()
