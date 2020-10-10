"""
This module is used to encrypt / decrypt the solution file.

All the functions are taken from this blog post:
https://eli.thegreenplace.net/2010/06/25/aes-encryption-of-files-in-python-with-pycrypto
https://github.com/eliben/code-for-blog/tree/master/2010/aes-encrypt-pycrypto

I also have made a change in the function to generate a key from a pass-phrase of arbitrary length. This is also based
on a suggestion made in the same post.

All credits to Eli Bendersky for the `encrypt_file` and `decrypt_file` functions.
"""
import os
import random
import struct
import hashlib
import sys
import getpass
from Crypto.Cipher import AES


def encrypt_file(key, in_filename, out_filename=None, chunksize=64*1024):
    """ Encrypts a file using AES (CBC mode) with the
        given key.
        key:
            The encryption key - a bytes object that must be
            either 16, 24 or 32 bytes long. Longer keys
            are more secure.
        in_filename:
            Name of the input file
        out_filename:
            If None, '<in_filename>.enc' will be used.
        chunksize:
            Sets the size of the chunk which the function
            uses to read and encrypt the file. Larger chunk
            sizes can be faster for some files and machines.
            chunksize must be divisible by 16.
    """
    if not out_filename:
        out_filename = in_filename + '.enc'

    iv = os.urandom(16)
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    filesize = os.path.getsize(in_filename)

    with open(in_filename, 'rb') as infile:
        with open(out_filename, 'wb') as outfile:
            outfile.write(struct.pack('<Q', filesize))
            outfile.write(iv)

            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                elif len(chunk) % 16 != 0:
                    chunk += b' ' * (16 - len(chunk) % 16)

                outfile.write(encryptor.encrypt(chunk))
    return out_filename


def decrypt_file(key, in_filename, out_filename=None, chunksize=24*1024):
    """ Decrypts a file using AES (CBC mode) with the
        given key. Parameters are similar to encrypt_file,
        with one difference: out_filename, if not supplied
        will be in_filename without its last extension
        (i.e. if in_filename is 'aaa.zip.enc' then
        out_filename will be 'aaa.zip')
    """
    if not out_filename:
        out_filename = os.path.splitext(in_filename)[0]

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        decryptor = AES.new(key, AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)
    return out_filename


if __name__ == '__main__':
    mode = "encrypt"
    if len(sys.argv) > 1:
        if sys.argv[1] == "-d":
            mode = "decrypt"
        else:
            raise ValueError("{} is not a recognized option".format(sys.argv[1]))
    passphrase = getpass.getpass()
    key = hashlib.sha256(passphrase.encode('utf-8')).digest()
    if mode == "encrypt":
        encrypt_file(key, "search.py", "search.py.enc")
        encrypt_file(key, "searchAgents.py", "searchAgents.py.enc")
    else:
        decrypt_file(key, "search.py.enc")
        decrypt_file(key, "searchAgents.py.enc")
