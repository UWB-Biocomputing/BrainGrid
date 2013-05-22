#ifndef _DELAYIDX_H_
#define _DELAYIDX_H_

//! Utility structure for tracking the time slot for a delayed queue
struct DelayIdx {
    //! The index indicating the current time slot in the delayed queue
    uint32_t delayIdx;
    //! length of the delayed queue in bits
    uint32_t ldelayQueue;

    DelayIdx( ) {
        delayIdx = 0;
        ldelayQueue = LENGTH_OF_DELAYQUEUE;
    }

    void inc( void ) {
        if ( ++delayIdx >= ldelayQueue )
            delayIdx = 0;
    }    

    uint32_t getBitmask( ) {    
        return ( 0x1 << delayIdx );
    }

    int getIndex( void ) {
        return delayIdx;
    }
};
#endif
