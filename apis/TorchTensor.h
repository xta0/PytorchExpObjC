//
//  TorchTensor.h
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, TorchTensorType) {
    TorchTensorTypeByte,
    TorchTensorTypeInt,
    TorchTensorTypeLong,
    TorchTensorTypeFloat,
    TorchTensorTypeUndefined,
};
//Tensors are immutable objects
@interface TorchTensor : NSObject<NSCopying>
//The type of the tensor
@property(nonatomic,assign, readonly) TorchTensorType type;
//The size of the self tensor. The returned value is a array of integer
@property(nonatomic,strong, readonly) NSArray<NSNumber* >* size;
//The number of dimensions of self tensor.
@property(nonatomic,assign, readonly) int64_t dim;
// Returns if a `Tensor` has quantized backend.
@property(nonatomic,assign, readonly) BOOL quantized;

//creat a tensor with a type, shape and a pointer to a buffer
+ (TorchTensor* )newWithType:(TorchTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* _Nullable)data;
+ (TorchTensor* )newWithType:(TorchTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* _Nullable)data Quantized:(BOOL) quantized;

@end

@interface TorchTensor(Operations)
//Performs Tensor dtype and/or device conversion.
- (TorchTensor* )to:(TorchTensorType) type;
//Get a number from a tensor containing a single value
- (NSNumber* )item;
//Permute the dimensions of this tensor.
- (TorchTensor* )permute:(NSArray<NSNumber* >*) dims;
//Returns a new tensor with the same data as the self tensor but of a different shape.
- (TorchTensor* )view:(NSArray<NSNumber* >*)size;

@end

@interface TorchTensor(ObjectSubscripting)

- (TorchTensor* )objectAtIndexedSubscript:(NSUInteger)idx;

@end

NS_ASSUME_NONNULL_END
