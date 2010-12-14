import _amplpy2 as a


if __name__ == '__main__':
    m = a.ampl('hs006')
    print 'n = ', m._n_var
    print 'm = ', m._n_con
    print m.get_x0()
    print m.get_pi0()
    print m.get_Lcon()
    print m.get_CType()
