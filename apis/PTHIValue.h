//
//  PTHIValue.h
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, PTHIValueType) {
    PTHIValueTypeTensor,
    PTHIValueTypeBool,
    PTHIValueTypeDouble,
    PTHIValueTypeInt,
    PTHIValueTypeBoolList,
    PTHIValueTypeIntList,
    PTHIValueTypeDoubleList,
    PTHIValueTypeTensorList,
};


@class PTHTensor;

@interface PTHIValue : NSObject

@property(nonatomic,assign, readonly) PTHIValueType type;

+ (instancetype) newIValueWithTensor:(PTHTensor* )tensor;
+ (instancetype) newIValueWithBool:(NSNumber* )value;
+ (instancetype) newIValueWithDouble:(NSNumber* )value;
+ (instancetype) newIValueWithInt:(NSNumber* )value;
+ (instancetype) newIValueWithBoolList:(NSArray<NSNumber*>* )value;
+ (instancetype) newIValueWithIntList:(NSArray<NSNumber*>* )value;
+ (instancetype) newIValueWithDoubleList:(NSArray<NSNumber*>* )value;
+ (instancetype) newIValueWithTensorList:(NSArray<PTHTensor*>* )value;

/** Unavailable. */
- (instancetype)init NS_UNAVAILABLE;

@end

@interface PTHIValue (Type)

- (PTHTensor* )toTensor;
- (NSNumber* )toBool;
- (NSNumber* )toInt;
- (NSNumber* )toDouble;
//- (NSArray<PTHIValue*>*) toTuple;
- (NSArray<NSNumber*>*)  toBoolList;
- (NSArray<NSNumber*>*)  toIntList;
- (NSArray<NSNumber*>*)  toDoubleList;
- (NSArray<PTHTensor*>*) toTensorList;

@end

NS_ASSUME_NONNULL_END
