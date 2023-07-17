
import sys, getopt
import argparse

# print("인자(argument) 개수:", len(sys.argv))
# print("인자(argument) 내용:", str(sys.argv))


def parse_args():
    parser = argparse.ArgumentParser(description='RIST')

    parser.add_argument('--part'         , type=str , default="3"   , help='')
    parser.add_argument('--answer' , type=str , default="100" , help='')
    parser.add_argument('--answer_range' , type=str , default="90"  , help='')
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    
    args = parse_args()
    # print(args.data_root)
    # print(args.search_range)
    # print(args.radius_range)