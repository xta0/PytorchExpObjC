//
//  TorchIValue+IValue.h
//  Pods-TestApp
//
//  Created by taox on 8/25/19.
//

#import "TorchIValue.h"
#import <PytorchExp/PytorchExp.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchIValue (Internal)

- (at::IValue )toIValue;

- (at::IValue* )unsafeImpl;

+ (TorchIValue* )newWithIValue:(const at::IValue& )value;

@end

NS_ASSUME_NONNULL_END
