/**
 *
 * @param date Date object to be parsed in "dd/mm/yyyy hh/MM/ss"
 * @returns Formatted date string
 */
export const getFormattedTimestamp = (date: Date | undefined): string => {
  if (date)
    return (
      date.getDate() +
      "/" +
      (date.getMonth() + 1) +
      "/" +
      date.getFullYear() +
      ", " +
      date.getHours() +
      ":" +
      date.getMinutes() +
      ":" +
      date.getSeconds()
    );
  else return "";
};
