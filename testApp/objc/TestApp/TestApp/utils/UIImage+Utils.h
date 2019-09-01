//
//  Copyright © 2019年 Vizlab. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <memory>

NS_ASSUME_NONNULL_BEGIN

@interface UIImage (Utils)


/**
 Resize the image with the given size

 @param sz The new size
 @return A new UIImage object with the given size
 */
- (UIImage* )resize:(CGSize)sz;

/**
 <#Description#>

 @return <#return value description#>
 */
- (float* )normalizedBuffer;

@end

NS_ASSUME_NONNULL_END
