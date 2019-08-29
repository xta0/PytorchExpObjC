# PytorchExpObjC

The Objective-C API wrapper of PytorchExp


## Quick start

Add the PytorchExpObjC to your `podfile`

- Objective-C

```
pod 'PytorchExpObjC', :git=>'https://github.com/xta0/PytorchExpObjC.git' ,:branch => 'master'
```
- Swift

```
use_framework!
pod 'PytorchExpObjC', :git=>'https://github.com/xta0/PytorchExpObjC.git' ,:branch => 'master'
```


## Note

Since Pytorch is not in its final stage, the static libraries in this pod hasn't been fully optimized yet and thus can't be used in production. For the demo purpose, the `libtorch.a` only supports one architecture - `arm64`, meaning the code can be only run on an arm64 device. And because of the size issues, the bitcode in `libtorch.a` has been stripped out, so `enable bitcode` switch in XCode should be turned off as well. 


## TestAPP

The TestApp folder contains an Objective-C demo, it uses a traced torch script model which was exported from torchvision (ResNet18). To run the demo, just simply run `pod install` 