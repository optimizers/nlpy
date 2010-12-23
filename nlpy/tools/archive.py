import gzip, bz2, tarfile

def tarxf(archive):
    """
    This (oddly named) function performs the same tas as the ``tar xf``
    command, i.e., extract a tar archive. The uncompressed archive can
    subsequently be found in the newly-created directory named ``archive``,
    where ``archive.tar.bz2`` is the name of the original compressed tar
    archive.
    """
    archivetar = tarfile.open(archive + '.tar')
    archivetar.extractall(path=archive)
    archivetar.close()
    return


def tarjxf(archive):
    """
    This (oddly named) function performs the same tas as the ``tar jxf``
    command, i.e., uncompress and extract a compressed tar archive all
    at once. The uncompressed archive can subsequently be found in the
    newly-created directory named ``archive``, where ``archive.tar.bz2``
    is the name of the original compressed tar archive.
    """
    archivetar_name = archive + '.tar'
    archivetarbz2_name = archivetar_name + '.bz2'

    # Uncompress into regular tar archive.
    archivetarbz2 = bz2.BZ2File(archivetarbz2_name, 'rb')
    archivetar = open(archivetar_name, mode='wb')
    for line in archivetarbz2:
        archivetar.write(line)
    archivetar.close()
    archivetarbz2.close()

    # Extract tar archive.
    tarxf(archive)

    return


def tarzxf(archive):
    """
    This (oddly named) function performs the same tas as the ``tar zxf``
    command, i.e., uncompress and extract a compressed tar archive all
    at once. The uncompressed archive can subsequently be found in the
    newly-created directory named ``archive``, where ``archive.tar.gz``
    is the name of the original compressed tar archive.
    """
    archivetar_name = archive + '.tar'
    archivetargz_name = archivetar_name + '.gz'

    # Uncompress into regular tar archive.
    archivetargz = gzip.GzipFile(archivetargz_name, 'rb')
    archivetar = open(archivetar_name, mode='wb')
    for line in archivetargz:
        archivetar.write(line)
    archivetar.close()
    archivetargz.close()

    # Extract tar archive.
    tarxf(archive)

    return


if __name__ == '__main__':
    from urllib import urlretrieve
    import os

    libampl_name = 'libampl'
    src = 'http://www.gerad.ca/~orban/LibAmpl/libampl.tar.bz2'

    # Fetch compressed tar archive.
    urlretrieve(src, filename=libampl_name + '.tar.bz2')

    # Uncompress and extract.
    tarjxf(libampl_name)

    # Change to unarchived directory and build.
    cwd = os.getcwd()
    os.chdir(libampl_name)
    os.system('make')
    os.chdir(cwd)
