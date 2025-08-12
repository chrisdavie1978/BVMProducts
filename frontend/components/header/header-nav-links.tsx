"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import cn from "classnames";

type headerLinkType = {
  href: string;
  label: string;
};
const headerLinks: Array<headerLinkType> = [];

const HeaderNavLinks = () => {
  const pathname = usePathname();

  return (
    <>
      {headerLinks.map(({ href, label }) => (
        <li key={`${href}${label}`} className="mx-5">
          <Link
            href={href}
            className={cn(
              `focus:text-accent flex items-center font-normal text-heading no-underline transition duration-200 hover:text-accent`,
              href === pathname ? "text-accent" : ""
            )}
          >
            {label}
          </Link>
        </li>
      ))}
    </>
  );
};

export default HeaderNavLinks;
