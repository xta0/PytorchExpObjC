//
//  PTHIValue+IValue.h
//  Pods-TestApp
//
//  Created by taox on 8/25/19.
//

#import "PTHIValue.h"
#import <PytorchExp/PytorchExp.h>

NS_ASSUME_NONNULL_BEGIN

@interface PTHIValue (Internal)

- (at::IValue )toIValue;

+ (PTHIValue* )newWithIValue:(const at::IValue& )value;

@end

NS_ASSUME_NONNULL_END
