#crooruhe
#caesar
def Caesar(s, k):
    result = ''

    for t in s:
        t = ord(t)
        if t > 126:
            t = t - 95
        elif t < 32:
            t = t + 95

        temp = t + k
        result += chr(temp)

    return result

def main():
    print(Caesar('abcdefg', 3))
    print(Caesar('defghij', -3))


if __name__ == "__main__":
    main()
