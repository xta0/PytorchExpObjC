//
//  TorchTensor+Internal.h
//  Pods-TestApp
//
//  Created by taox on 8/26/19.
//

#import "TorchTensor.h"
#import <PytorchExp/PytorchExp.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchTensor (Internal)

- (at::Tensor)toTensor;
+ (TorchTensor* )newWithTensor:(const at::Tensor& ) tensor;

@end

NS_ASSUME_NONNULL_END
