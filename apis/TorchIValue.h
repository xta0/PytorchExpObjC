//
//  TorchIValue.h
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, TorchIValueType) {
    TorchIValueTypeTensor,
    TorchIValueTypeBool,
    TorchIValueTypeDouble,
    TorchIValueTypeInt,
    TorchIValueTypeBoolList,
    TorchIValueTypeIntList,
    TorchIValueTypeDoubleList,
    TorchIValueTypeTensorList,
};

@class TorchTensor;
@interface TorchIValue : NSObject

@property(nonatomic,assign, readonly) TorchIValueType type;

+ (instancetype) newWithTensor:(TorchTensor* )tensor;
+ (instancetype) newWithBool:(NSNumber* )value;
+ (instancetype) newWithDouble:(NSNumber* )value;
+ (instancetype) newWithInt:(NSNumber* )value;
+ (instancetype) newWithBoolList:(NSArray<NSNumber*>* )value;
+ (instancetype) newWithIntList:(NSArray<NSNumber*>* )value;
+ (instancetype) newWithDoubleList:(NSArray<NSNumber*>* )value;
+ (instancetype) newWithTensorList:(NSArray<TorchTensor*>* )value;

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

@end

@interface TorchIValue (Type)

- (TorchTensor* )toTensor;
- (NSNumber* )toBool;
- (NSNumber* )toInt;
- (NSNumber* )toDouble;
- (NSArray<NSNumber*>*)  toBoolList;
- (NSArray<NSNumber*>*)  toIntList;
- (NSArray<NSNumber*>*)  toDoubleList;
- (NSArray<TorchTensor*>*) toTensorList;

@end

NS_ASSUME_NONNULL_END
