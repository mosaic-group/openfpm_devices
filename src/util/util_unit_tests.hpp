/*
 * util_unit_tests.hpp
 *
 *  Created on: Oct 2, 2019
 *      Author: i-bird
 */

#ifndef UTIL_UNIT_TESTS_HPP_
#define UTIL_UNIT_TESTS_HPP_

BOOST_AUTO_TEST_SUITE( util_test_test )

BOOST_AUTO_TEST_CASE( align_number_test )
{
	BOOST_REQUIRE_EQUAL(align_number(8,3),8);
	BOOST_REQUIRE_EQUAL(align_number(8,9),16);

	BOOST_REQUIRE_EQUAL(align_number(3,3),3);
	BOOST_REQUIRE_EQUAL(align_number(3,7),9);
}

BOOST_AUTO_TEST_SUITE_END()


#endif /* UTIL_UNIT_TESTS_HPP_ */
