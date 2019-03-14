# Img Crypt

Scripts for testing different image encryption methods. Both standard stream ciphers and dedicated algorithm are implemented.

### [imgcrypt.py](../master/img_crypt/imgcrypt.py)

Chosen image is being encrypted and decrypted. When procedure finishes script
displays image before and after encryption, along with histograms of their respective
pixel values.

### [speed_test.py](../master/img_crypt/speed_test.py)

Script encrypts and decrypts sample image and prints
time it took for each scheme available. 


## Usage

### imgcrypt.py

```imgcrypt.py [-h] [-s SCHEME] [-l LOCATOR] [-e EXAMPLE]```

* ```-h, --help```            prints argparse generated help
* ```-s SCHEME, --scheme SCHEME``` specify encryption scheme. Allowed schemes: ```3DES_ECB,
3DES_CBC, AES_ECB, AES_CBC, CML, CML_MULTI``` Not case sensitive.
* ```  -l LOCATOR, --locator LOCATOR ``` specify image to be encrypted, can be path to the local file or url of an image hosted on internet
* ```-e EXAMPLE, --example EXAMPLE``` Name of a example picture for hardcoded url in ```example_urls``` variable. Example picture names: ```obelix ,matterhorn, cat, balloon```.

#### Examples

```python imgcrypt.py -l /path/to/image.png -s 3DES_ECB``` 

Encrypt image saved on hard drive with 3DES in ECB mode.

```python imgcrypt.py -l https://example.com/images/image.jpg -s AES_CBC``` 

Encrypt image referenced by url with AES in CBC mode.

```python imgcrypt.py -e obelix -s CML_MULTI``` 

Encrypt specified example image with CML algorithm using multiprocessing.

### speedtest.py

Script takes no parameters since it uses example picture and all available encryption schemes. 

## Encryption schemes

### Standard encryption schemes

* AES in CBC, ECB modes ```AES_CBC, AES_ECB```
* 3DES in CBC, ECB modes ```AES_CBC, AES_ECB```

### Dedicated encryption scheme

* Image encryption algorithm based on chaotic map lattices ```CML```, ```CML_MULTI```

For more info on algorithm see [cml.py](../master/img_crypt/cml.py) docstring.

This scheme is much slower than standard ciphers. ```CML_MULTI``` uses multiprocessing module to perform faster encryption/decryption but can still can take up to a minute on bigger pictures.

## Requirements

Python 3.6+ with modules: pillow, cryptography, numpy, matplotlib

## Based on 
* On improved image encryption scheme based on chaotic map lattices (K. Jastrzębski, Z. Kotulski 2009)
