//
//  PTHTensor.h
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, PTHTensorType) {
    PTHTensorTypeByte,
    PTHTensorTypeInt,//32bit integer
    PTHTensorTypeLong,//64bit integer
    PTHTensorTypeFloat, //32bit float
    PTHTensorTypeUndefined,
};

// Tensors are immutable objects
@interface PTHTensor : NSObject<NSCopying>
//The type of the tensor
@property(nonatomic,assign, readonly) PTHTensorType type;
//The size of the self tensor. The returned value is a array of integer
@property(nonatomic,strong, readonly) NSArray<NSNumber* >* size;
//The number of dimensions of self tensor.
@property(nonatomic,assign, readonly) int64_t dim;
@property(nonatomic,assign, readonly) int64_t capacity;
@property(nonatomic,assign, readonly) BOOL quantized;

//creat a tensor with a type, shape and a pointer to a buffer
+ (PTHTensor* )newWithType:(PTHTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* )data;
+ (PTHTensor* )newWithType:(PTHTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* )data Quantized:(BOOL) quantized;

@end

@interface PTHTensor(Operations)
//Performs Tensor dtype and/or device conversion.
- (PTHTensor* )to:(PTHTensorType) type;
//Get a number from a tensor containing a single value
- (NSNumber* )item;
//Permute the dimensions of this tensor.
- (PTHTensor* )permute:(NSArray<NSNumber* >*) dims;
//mutable version apis (probably will be deleted)
- (PTHTensor* )add_:(float) value;
- (PTHTensor* )sub_:(float) value;
- (PTHTensor* )mul_:(float) value;
- (PTHTensor* )div_:(float) value;
/*
 Returns the k largest elements of the given input tensor along a given dimension.
 If largest is False then the k smallest elements are returned.
 A namedtuple of (values, indices) is returned, where the indices are the indices of the elements in the original input tensor.
 The boolean option sorted if True, will make sure that the returned k elements are themselves sorted
 */
- (NSArray<PTHTensor* >* )topKResult:(NSNumber* )k
                                  Dim:(NSNumber* )dim
                            isLargest:(bool) largest
                             isSorted:(bool) sorted;
//Returns a new tensor with the same data as the self tensor but of a different shape.
- (PTHTensor* )view:(NSArray<NSNumber* >*)v;

@end

@interface PTHTensor(ObjectSubscripting)

- (PTHTensor* )objectAtIndexedSubscript:(NSUInteger)idx;

@end

NS_ASSUME_NONNULL_END
