from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import binascii


class AESECBPKCS5Padding:
    def __init__(self, key, out_format="hex"):
        self.key = key.encode('utf-8') if isinstance(key, str) else key
        self.out_format = out_format

    def _pkcs5_pad(self, data):
        pad_len = 16 - (len(data) % 16)
        return data + bytes([pad_len] * pad_len)

    def _pkcs5_unpad(self, data):
        pad_len = data[-1]
        return data[:-pad_len]

    def encrypt(self, plaintext):
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        padded = self._pkcs5_pad(plaintext)
        cipher = Cipher(algorithms.AES(self.key), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        if self.out_format == 'hex':
            return ciphertext.hex()
        elif self.out_format in ('base64', 'b64'):
            import base64
            return base64.b64encode(ciphertext).decode('utf-8')
        return ciphertext

    def decrypt(self, ciphertext):
        if self.out_format == 'hex':
            ciphertext = bytes.fromhex(ciphertext)
        elif self.out_format in ('base64', 'b64'):
            import base64
            ciphertext = base64.b64decode(ciphertext)
        cipher = Cipher(algorithms.AES(self.key), modes.ECB(), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        unpadded = self._pkcs5_unpad(plaintext)
        return unpadded.decode('utf-8')
