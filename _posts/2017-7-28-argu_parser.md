---
layout: post
comments: true
categories: diary
---
## Title

#!/usr/bin/python

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="arg_test Parsers argument test")
    parser.add_argument("-i","--int", type=str, action="store", help="imageP-th")
    parser.add_argument("-s","--str", type=str, action="store", help="imageP-th")    
    parser.add_argument("-sf", "--store_false", default=True, action="store_false", help="Save DB")
    parser.add_argument("-st", "--store_true", default=False, action="store_true", help="Save DB")    
    args = parser.parse_args()    

    print(type(args.int))
    print(args.int)

    print(type(args.str))
    print(args.str)  

    print(type(args.store_false))
    print(args.store_false)

    print(type(args.store_true))
    print(args.store_true)    
