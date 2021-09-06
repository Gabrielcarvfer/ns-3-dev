/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (C) 2021 Universidade de Brasília
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#ifndef FORCE_INLINE_H
#define FORCE_INLINE_H

#ifdef NS3_ENABLE_FORCE_INLINE
    #ifndef _MSC_VER
        #define NS3_INLINE __attribute__((always_inline)) inline
    #else
        #define NS3_INLINE __forceinline
    #endif
#else
    #define NS3_INLINE inline
#endif

#endif //FORCE_INLINE_H
