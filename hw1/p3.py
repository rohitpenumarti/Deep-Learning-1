from func import f
import sys

def secant_method(g, a, b, tol=1e-10):
    x_n1 = a
    x_n2 = b
    x_n = x_n1 - g(x_n1)*((x_n2-x_n1)/(g(x_n2)-g(x_n1)))
    N = 1

    while abs(x_n-x_n1) >= tol:
        x_n2 = x_n1
        x_n1 = x_n
        N += 1
        x_n = x_n1 - g(x_n1)*((x_n2-x_n1)/(g(x_n2)-g(x_n1)))

    return x_n, x_n1, x_n2, N

def isfloat(s):
    try:
        float(s)
    except Exception:
        return False

    return True

def main():
    if len(sys.argv) == 3:
        in1, in2 = sys.argv[1:3]
        if isfloat(in1) and isfloat(in2):
            a = float(in1)
            b = float(in2)
            if (a < b) and (f(a)*f(b) < 0):
                x_N, x_N1, x_N2, N = secant_method(f, a, b)
                res = str(N) + "\n" + str(x_N2) + "\n" + str(x_N1) + "\n" + str(x_N)
                sys.stdout.write(str(res))
            else:
                sys.stderr.write("Range error")
        else:
            sys.stderr.write("Range error")

if __name__ == "__main__":
    main()