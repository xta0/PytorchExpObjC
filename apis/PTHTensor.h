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
    PTHTensorTypeInt,
    PTHTensorTypeFloat,
    PTHTensorTypeUndefined,
};


// Tensors are immutable objects
@interface PTHTensor : NSObject

@property(nonatomic,assign, readonly) PTHTensorType type;
@property(nonatomic,strong, readonly) NSArray<NSNumber* >* shape;
@property(nonatomic,assign, readonly) int64_t dim;


+ (PTHTensor* )newWithType:(PTHTensorType)type Shape:(NSArray<NSNumber* >*)dims Data:(void* )data;

@end

@interface PTHTensor(Operations)

- (PTHTensor* )toType:(PTHTensorType) type;

- (PTHTensor* )permute:(NSArray<NSNumber* >*) dims;

@end

NS_ASSUME_NONNULL_END
