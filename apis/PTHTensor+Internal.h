//
//  PTHTensor+Internal.h
//  Pods-TestApp
//
//  Created by taox on 8/26/19.
//

#import "PTHTensor.h"
#import <PytorchExp/PytorchExp.h>

NS_ASSUME_NONNULL_BEGIN

@interface PTHTensor (Internal)

- (at::Tensor)toTensor;

- (at::Tensor* )unsafeImpl;

+ (PTHTensor* )newWithTensor:(const at::Tensor& ) tensor;

@end

NS_ASSUME_NONNULL_END
