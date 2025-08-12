import cn from "classnames";
import HeaderNavLinks from "@/components/header/header-nav-links";
import Logo from "@/components/header/logo";

const Header = () => {
  return (
    <header className={cn("site-header-with-search h-14 md:h-18 lg:h-22")}>
      <div
        className={cn(
          `fixed z-50 flex h-14 w-full transform-gpu items-center justify-between border-b border-gray-200 bg-white px-5 py-2 shadow-sm transition-transform duration-300 md:h-18 lg:h-22 lg:px-8`
        )}
      >
        <div className="flex w-full items-center lg:w-auto">
          <Logo classNames="mx-auto lg:mx-0" />
        </div>
        <ul className="hidden shrink-0 items-center space-x-10 rtl:space-x-reverse lg:flex">
          <HeaderNavLinks />
        </ul>
      </div>
    </header>
  );
};

export default Header;
