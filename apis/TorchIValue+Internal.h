#import "TorchIValue.h"
#import <PytorchExp/PytorchExp.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchIValue (Internal)

- (at::IValue )toIValue;
+ (TorchIValue* )newWithIValue:(const at::IValue& )value;

@end

NS_ASSUME_NONNULL_END
